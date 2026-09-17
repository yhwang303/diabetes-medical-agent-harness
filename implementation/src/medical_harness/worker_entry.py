"""Private child entry point. JSON only; no ledger, service token or arbitrary import commands."""

import os
import errno
import resource
import signal
import sys
from functools import partial
from threading import Thread

from .adapters import REAL_EXECUTORS, basal_fixture_registry, legacy_basal_fixture_registry, fixture_registry, policy_scenario_registry
from .contracts import GateError, canonical, strict_json
from .sdk_worker import REPORT_EXECUTOR, REVIEW_EXECUTORS, SDK_EXECUTOR, PREDICTION_AGENT_EXECUTOR, RL_AGENT_EXECUTOR


def registry(timeout):
    return {x.identity: x for x in [*fixture_registry().values(), *basal_fixture_registry().values(), *legacy_basal_fixture_registry().values(), *policy_scenario_registry(timeout).values(),
                                     *REAL_EXECUTORS.values(), SDK_EXECUTOR, REPORT_EXECUTOR, PREDICTION_AGENT_EXECUTOR, RL_AGENT_EXECUTOR, *REVIEW_EXECUTORS.values()]}


def registered_executor(executor, timeout):
    value = registry(timeout).get(executor.identity)
    if value is None or value.version != executor.version or value.origin != executor.origin:
        raise GateError('WORKER_EXECUTOR_NOT_REGISTERED', 503)
    left, right = executor.call, value.call
    if isinstance(left, partial) and isinstance(right, partial):
        # Scenario delay was frozen when Core was initialized, not derived from remaining job time.
        matches = (left.func is right.func and left.args == right.args
                   and set(left.keywords) == set(right.keywords) == {'scenario', 'timeout'}
                   and left.keywords['scenario'] == right.keywords['scenario'])
    else:
        matches = left is right
    if not matches:
        raise GateError('WORKER_EXECUTOR_NOT_REGISTERED', 503)
    return value


def parent_watch(life_fd):
    def watch():
        try:
            os.read(life_fd, 1)
        finally:
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGKILL)
            os._exit(1)
    Thread(target=watch, daemon=True, name='parent-liveness').start()


def set_limits(limits):
    os.umask(0o077)
    for kind, soft, hard in [(resource.RLIMIT_CORE, 0, 0),
        (resource.RLIMIT_CPU, limits['cpu_seconds'], limits['cpu_seconds'] + 1),
        (resource.RLIMIT_FSIZE, limits['file_bytes'], limits['file_bytes']),
        (resource.RLIMIT_NOFILE, limits['open_files'], limits['open_files'])]:
        resource.setrlimit(kind, (soft, hard))


def main():
    parent_watch(int(sys.argv[1]))
    try:
        envelope = strict_json(sys.stdin.buffer.read(131073))
        try:
            set_limits(envelope['limits'])
        except (OSError, ValueError):
            os._exit(78)
        executor = registry(envelope['timeout']).get(envelope['executor'])
        if executor is None or executor.version != envelope['version'] or executor.origin != envelope['origin']:
            raise GateError('WORKER_EXECUTOR_NOT_REGISTERED')
        result = canonical({'ok': True, 'data': executor.call(envelope['request'])}).encode()
        if len(result) > envelope['limits']['output_bytes']:
            os._exit(74)
        sys.stdout.buffer.write(result)
        sys.stdout.buffer.flush()
    except OSError as exc:
        os._exit(76 if exc.errno == errno.EFBIG else 77 if exc.errno == errno.EMFILE else 70)
    except MemoryError:
        os._exit(75)
    except BaseException:
        # Exceptions, model prose and credentials never become the IPC response or service log.
        os._exit(70)


if __name__ == '__main__':
    main()
