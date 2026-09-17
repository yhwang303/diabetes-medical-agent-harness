import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from medical_harness.adapters import fixture_registry
from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.store import Store
from medical_harness.workers import WorkerLimits, WorkerRunner


class ProbeRunner(WorkerRunner):
    def __init__(self, probe, path, **kwargs):
        super().__init__(**kwargs)
        self.probe, self.path = probe, path

    def _command(self, life_fd):
        return [sys.executable, '-I', str(Path(__file__).with_name('worker_probe.py')), str(life_fd), self.probe, str(self.path)]


def alive(pid):
    # An adopted zombie may briefly remain on macOS; it cannot run and must be reaped by init.
    row = subprocess.run(['/bin/ps','-o','stat=','-p',str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(row) and not row.startswith('Z')


def invoke(runner, *, timeout=2, notify=lambda *a, **kw: None, key='job'):
    return runner.run(fixture_registry()['prediction'], {}, timeout=timeout, validate=lambda:None, job_id=key, notify=notify)


@pytest.mark.parametrize('probe,code', [('hang','EXECUTOR_TIMEOUT'), ('crash','WORKER_CRASHED'),
    ('cpu','WORKER_CPU_LIMIT'), ('memory','WORKER_MEMORY_LIMIT'), ('output','WORKER_OUTPUT_LIMIT'), ('malformed','WORKER_PROTOCOL_ERROR')])
def test_actual_child_faults_terminate_and_release_capacity(tmp_path, probe, code):
    path = tmp_path / 'probe.json'
    runner = ProbeRunner(probe, path, limits=WorkerLimits(cpu_seconds=1, memory_bytes=64*1024*1024), max_workers=1)
    events = []
    try:
        with pytest.raises(GateError, match=code):
            invoke(runner, timeout=1 if probe=='hang' else 4, notify=lambda kind, **data:events.append((kind,data)))
        pid = next(data['pid'] for kind,data in events if kind=='worker_started')
        assert not alive(pid) and not runner._active
        assert events[-1][1]['reason'] == code
        runner.probe = 'limits'
        assert invoke(runner)['file_limited']  # Capacity is usable after the failed process is reaped.
    finally:
        runner.close()


def test_os_limits_and_clean_environment_are_effective(tmp_path, monkeypatch):
    monkeypatch.setenv('PRIVATE_TEST_CREDENTIAL', 'must-not-reach-child')
    runner = ProbeRunner('limits', tmp_path / 'limits.json')
    try:
        result = invoke(runner)
        assert result['pid'] != os.getpid() and result['ppid'] == os.getpid()
        assert result['cpu'] == [2,3] and result['core'] == [0,0]
        assert result['file_limited'] and result['fd_limited']
        assert 'PRIVATE_TEST_CREDENTIAL' not in result['env_keys']
        assert result['cwd'].endswith('/implementation/runtime/workers')
        assert (tmp_path / 'limits.large').stat().st_size <= runner.limits.file_bytes
        assert not alive(result['pid'])
    finally:
        runner.close()


def test_timeout_kills_process_group_descendant(tmp_path):
    path = tmp_path / 'children.json'
    runner = ProbeRunner('descendant', path)
    try:
        with pytest.raises(GateError, match='EXECUTOR_TIMEOUT'):
            invoke(runner, timeout=1)
        pids = json.loads(path.read_text())
        assert not alive(pids['pid']) and not alive(pids['child'])
    finally:
        runner.close()


def test_concurrency_and_shutdown_reap_active_processes(tmp_path):
    runner = ProbeRunner('hang', tmp_path / 'shutdown.json', max_workers=1)
    started = Event()
    events = []
    def notify(kind, **metadata):
        events.append((kind,metadata))
        if kind=='worker_started': started.set()
    with ThreadPoolExecutor() as pool:
        task = pool.submit(invoke, runner, timeout=10, notify=notify)
        assert started.wait(2)
        with pytest.raises(GateError, match='EXECUTOR_BUSY'):
            invoke(runner, key='another')
        runner.close()
        with pytest.raises(GateError, match='CORE_SHUTTING_DOWN'):
            task.result(timeout=2)
    assert not alive(events[0][1]['pid']) and not runner._active
    with pytest.raises(GateError, match='CORE_SHUTTING_DOWN'):
        invoke(runner)


def test_shutdown_does_not_signal_an_already_reaped_group(tmp_path, monkeypatch):
    import medical_harness.workers as workers
    original, signaled = workers.kill_group, set()
    def reject_repeated_signal(process):
        # Reproduce Darwin refusing a second killpg during concurrent shutdown/finally.
        if process.pid in signaled:
            raise PermissionError('group already reaped')
        signaled.add(process.pid)
        return original(process)
    monkeypatch.setattr(workers, 'kill_group', reject_repeated_signal)
    runner = ProbeRunner('hang', tmp_path / 'reaped.json')
    started = Event()
    try:
        with ThreadPoolExecutor() as pool:
            task = pool.submit(invoke, runner, timeout=10,
                               notify=lambda kind, **data: started.set() if kind == 'worker_started' else None)
            assert started.wait(2)
            runner.close()
            with pytest.raises(GateError, match='CORE_SHUTTING_DOWN'):
                task.result(timeout=2)
        assert len(signaled) == 1 and all(not alive(pid) for pid in signaled)
        assert not runner._active
    finally:
        runner.close()


@pytest.mark.parametrize('action,reason', [('cancel','CANCELLED'), ('snapshot','SNAPSHOT_CHANGED'), ('revoke','VERSION_REVOKED')])
def test_core_invalidations_terminate_real_workers_before_result(tmp_path, snapshot, action, reason):
    runner = ProbeRunner('hang', tmp_path / 'pending.json')
    core = Core(Store(tmp_path / 'core.sqlite3'), enable_fixtures=True, executor_runner=runner)
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'eval', 'pending')
    try:
        with ThreadPoolExecutor() as pool:
            task = pool.submit(core.execute_prediction, 'alice', run['id'])
            deadline = time.monotonic()+2
            while not runner.path.exists() and time.monotonic()<deadline: time.sleep(.01)
            assert runner.path.exists()
            pid = json.loads(runner.path.read_text())['pid']
            if action=='cancel': core.cancel('alice', run['id'])
            elif action=='snapshot': core.update_case('alice', case['case_id'], snapshot)
            else: core.revoke_version('fixture-prediction-v1')
            with pytest.raises(GateError, match=reason): task.result(timeout=2)
            assert not alive(pid)
        with core.store.tx() as db:
            assert db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0]==0
            assert db.execute('SELECT status FROM jobs').fetchone()[0]=='FENCED'
    finally:
        core.close()


@pytest.mark.parametrize('scenario,reason', [('candidate',None), ('abstain','POLICY_ABSTAIN'), ('unsupported','POLICY_UNSUPPORTED'),
    ('error','POLICY_ERROR'), ('invalid_output','INVALID_MODEL_OUTPUT'), ('parent_mismatch','FORECAST_PARENT_MISMATCH'),
    ('safety_rejected','ENGINEERING_SAFETY_REJECTED'), ('timeout','EXECUTOR_TIMEOUT')])
def test_production_process_registry_and_core_gates(tmp_path, snapshot, scenario, reason):
    core = Core(Store(tmp_path / 'production.sqlite3'), enable_fixtures=True, executor_timeout=1)
    try:
        case = core.create_case('alice', snapshot)
        run = core.start_run('alice', case['case_id'], 'eval', 'scenario', case['snapshot_id'], scenario)
        if reason and scenario!='safety_rejected':
            with pytest.raises(GateError, match=reason): core.execute_models('alice', run['id'])
        else: core.execute_models('alice', run['id'])
        assert core.status('alice', run['id'])['reason']==reason
        assert not core.case_evidence('alice', case['case_id'])['release_allowed']
        events = core.events('alice', run['id'])
        with core.store.tx() as db:
            starts=[json.loads(r['body']) for r in db.execute("SELECT body FROM events WHERE kind='worker_started'")]
            stops=[json.loads(r['body']) for r in db.execute("SELECT body FROM events WHERE kind='worker_stopped'")]
            assert len(starts)==len(stops)==2
            assert all(not alive(e['pid']) for e in starts)
            assert db.execute('SELECT COUNT(*) FROM releases').fetchone()[0]==0
            if scenario=='timeout': assert db.execute("SELECT COUNT(*) FROM artifacts WHERE kind='policy'").fetchone()[0]==0
        assert 'payload' not in canonical(events) and 'action_value' not in canonical(events)
    finally:
        core.close()


def test_unregistered_callable_never_executes_in_core(tmp_path):
    runner = WorkerRunner()
    executor = replace(fixture_registry()['prediction'], call=lambda request: pytest.fail('unsafe fallback'))
    try:
        with pytest.raises(GateError, match='WORKER_EXECUTOR_NOT_REGISTERED'):
            runner.run(executor, {}, timeout=1, validate=lambda:None, job_id='unsafe', notify=lambda *a,**k:None)
    finally:
        runner.close()


def test_parent_sigkill_causes_worker_and_descendant_to_exit(tmp_path):
    path = tmp_path / 'orphan.json'
    parent = subprocess.Popen([sys.executable, '-I', str(Path(__file__).with_name('worker_parent_probe.py')), str(path)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pids = {}
    try:
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            try: pids = json.loads(path.read_text())
            except (OSError, ValueError): pass
            if 'child' in pids: break
            time.sleep(.01)
        assert 'child' in pids and alive(pids['pid']) and alive(pids['child'])
        parent.kill()
        parent.wait(timeout=2)
        deadline = time.monotonic()+2
        while time.monotonic()<deadline and any(alive(pid) for pid in pids.values()): time.sleep(.01)
        assert all(not alive(pid) for pid in pids.values())
    finally:
        if parent.poll() is None: parent.kill()
        parent.wait(timeout=2)
        # Clean test-only processes if an assertion failed, without concealing that failure.
        for pid in pids.values():
            if alive(pid): os.kill(pid, signal.SIGKILL)


def test_monitor_failure_is_fail_closed(tmp_path, monkeypatch):
    runner = ProbeRunner('hang', tmp_path/'monitor.json')
    events = []
    def fail(pid): raise GateError('WORKER_MONITOR_UNAVAILABLE', 503)
    monkeypatch.setattr('medical_harness.workers.group_rss', fail)
    try:
        with pytest.raises(GateError, match='WORKER_MONITOR_UNAVAILABLE'):
            invoke(runner, notify=lambda kind,**metadata:events.append((kind,metadata)))
        assert not alive(events[0][1]['pid']) and not runner._active
    finally:
        runner.close()


def test_worker_audit_failure_cannot_register_completed_output(tmp_path, snapshot, monkeypatch):
    import sqlite3
    core = Core(Store(tmp_path/'audit.sqlite3'), enable_fixtures=True)
    event = core._event
    def fail(db, run_id, kind, **metadata):
        if kind=='worker_stopped': raise sqlite3.OperationalError('audit unavailable')
        return event(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    try:
        case = core.create_case('alice', snapshot)
        run = core.start_run('alice', case['case_id'], 'eval', 'audit')
        with pytest.raises(GateError, match='EXECUTOR_ERROR'): core.execute_prediction('alice', run['id'])
        assert not core._runner._active
        with core.store.tx() as db:
            assert db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0]==0
    finally:
        core.close()
