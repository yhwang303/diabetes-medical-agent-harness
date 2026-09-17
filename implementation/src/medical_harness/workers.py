"""Bounded, disposable executor processes. Only Core owns the ledger and acceptance."""

import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from threading import BoundedSemaphore, Lock

from .contracts import GateError, canonical, strict_json
from .paths import ROOT, confined


@dataclass(frozen=True)
class WorkerLimits:
    cpu_seconds: int = 2
    memory_bytes: int = 256 * 1024 * 1024
    file_bytes: int = 1024 * 1024
    open_files: int = 64
    output_bytes: int = 131072


def kill_group(process):
    # Run even after the leader exited: ordinary descendants must not outlive a job.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def group_rss(pid):
    # Darwin's RSS rlimit is not used as a hard memory promise. Monitor the process group.
    try:
        result = subprocess.run(['/bin/ps', '-axo', 'pgid=,rss='], capture_output=True,
                                timeout=0.5, check=True, env={'PATH': '/usr/bin:/bin'})
        return sum(int(rss) * 1024 for group, rss in
                   (line.split() for line in result.stdout.decode('ascii').splitlines()) if int(group) == pid)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise GateError('WORKER_MONITOR_UNAVAILABLE', 503) from exc


class WorkerRunner:
    def __init__(self, *, limits=None, max_workers=4):
        self.limits = limits or WorkerLimits()
        self._capacity = BoundedSemaphore(max_workers)
        self._lock = Lock()
        self._active = {}
        self._closed = False

    def _command(self, life_fd):
        return [sys.executable, '-I', '-B', '-m', 'medical_harness.worker_entry', str(life_fd)]

    def close(self):
        with self._lock:
            self._closed = True
            for job_id, process in list(self._active.items()):
                kill_group(process)
                self._active.pop(job_id)

    def run(self, executor, request, *, timeout, validate, job_id, notify):
        from .worker_entry import registered_executor
        # Do not silently execute a replaced Python callable in Core or serialize Python objects.
        registered_executor(executor, timeout)
        if not self._capacity.acquire(blocking=False):
            raise GateError('EXECUTOR_BUSY', 503)
        process = None
        life_read = life_write = None
        reason, peak = 'completed', 0
        started = time.monotonic()
        try:
            validate()
            envelope = {'executor': executor.identity, 'version': executor.version, 'origin': executor.origin,
                        'timeout': timeout, 'request': request, 'limits': asdict(self.limits)}
            data = canonical(envelope).encode()
            if len(data) > 131072:
                raise GateError('WORKER_INPUT_LIMIT', 422)
            workdir = confined(ROOT / 'runtime/workers')
            workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
            life_read, life_write = os.pipe()
            with self._lock:
                if self._closed:
                    raise GateError('CORE_SHUTTING_DOWN', 503)
                process = subprocess.Popen(self._command(life_read), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, cwd=workdir, start_new_session=True, close_fds=True,
                    pass_fds=(life_read,), env={'PATH':'/usr/bin:/bin', 'LANG':'en_US.UTF-8',
                                               'TMPDIR':str(workdir), 'PYTHONDONTWRITEBYTECODE':'1'})
                self._active[job_id] = process
            os.close(life_read)
            life_read = None
            notify('worker_started', pid=process.pid, limits=asdict(self.limits))
            output = bytearray()
            position = 0
            deadline, next_check = started + timeout, 0
            with selectors.DefaultSelector() as selector:
                for stream, event in [(process.stdin, selectors.EVENT_WRITE), (process.stdout, selectors.EVENT_READ)]:
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, event)
                while selector.get_map():
                    now = time.monotonic()
                    if self._closed:
                        raise GateError('CORE_SHUTTING_DOWN', 503)
                    if now >= deadline:
                        raise GateError('EXECUTOR_TIMEOUT', 503)
                    if now >= next_check:
                        validate()
                        peak = max(peak, group_rss(process.pid))
                        if peak > self.limits.memory_bytes:
                            raise GateError('WORKER_MEMORY_LIMIT', 503)
                        next_check = time.monotonic() + 0.05
                    for key, _ in selector.select(min(0.025, max(0, deadline - time.monotonic()))):
                        if key.fileobj is process.stdin:
                            try:
                                position += os.write(process.stdin.fileno(), data[position:position + 8192])
                            except BrokenPipeError:
                                position = len(data)
                            if position == len(data):
                                selector.unregister(process.stdin)
                                process.stdin.close()
                        else:
                            chunk = os.read(process.stdout.fileno(), 8192)
                            if not chunk:
                                selector.unregister(process.stdout)
                            else:
                                output.extend(chunk)
                                if len(output) > self.limits.output_bytes:
                                    raise GateError('WORKER_OUTPUT_LIMIT', 503)
                # A process that closes stdout then hangs is still bounded by the same deadline.
                while process.poll() is None:
                    validate()
                    if self._closed:
                        raise GateError('CORE_SHUTTING_DOWN', 503)
                    if time.monotonic() >= deadline:
                        raise GateError('EXECUTOR_TIMEOUT', 503)
                    peak = max(peak, group_rss(process.pid))
                    if peak > self.limits.memory_bytes:
                        raise GateError('WORKER_MEMORY_LIMIT', 503)
                    time.sleep(0.01)
            validate()
            if self._closed:
                raise GateError('CORE_SHUTTING_DOWN', 503)
            if process.returncode != 0:
                code = {-signal.SIGXCPU:'WORKER_CPU_LIMIT', 74:'WORKER_OUTPUT_LIMIT', 75:'WORKER_MEMORY_LIMIT',
                        76:'WORKER_FILE_LIMIT', 77:'WORKER_FD_LIMIT', 78:'WORKER_LIMITS_UNAVAILABLE'}.get(process.returncode, 'WORKER_CRASHED')
                raise GateError(code, 503)
            try:
                result = strict_json(bytes(output))
                if not isinstance(result, dict) or set(result) != {'ok', 'data'} or result['ok'] is not True:
                    raise ValueError('worker protocol')
                return result['data']
            except (ValueError, GateError) as exc:
                raise GateError('WORKER_PROTOCOL_ERROR', 503) from exc
        except GateError as exc:
            reason = exc.code
            raise
        except Exception:
            reason = 'WORKER_FAILED'
            raise
        finally:
            try:
                if process is not None:
                    # close() and task cleanup share ownership of termination under this lock.
                    # Never signal a group that close() has already killed and reaped.
                    with self._lock:
                        if job_id in self._active:
                            kill_group(process)
                            self._active.pop(job_id)
                    for stream in (process.stdin, process.stdout):
                        stream.close()
                if life_read is not None:
                    os.close(life_read)
                if life_write is not None:
                    os.close(life_write)
                if process is not None:
                    notify('worker_stopped', pid=process.pid, reason=reason, exit_code=process.returncode,
                           elapsed_seconds=round(time.monotonic()-started, 4), peak_rss_bytes=peak)
            finally:
                self._capacity.release()
