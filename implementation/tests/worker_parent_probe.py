"""Disposable parent for testing actual parent death, not cooperative shutdown."""

import sys
from pathlib import Path

from medical_harness.adapters import fixture_registry
from medical_harness.workers import WorkerRunner


class ParentRunner(WorkerRunner):
    def _command(self, life_fd):
        return [sys.executable, '-I', str(Path(__file__).with_name('worker_probe.py')), str(life_fd), 'descendant', sys.argv[1]]


ParentRunner().run(fixture_registry()['prediction'], {}, timeout=30, validate=lambda:None,
                   job_id='parent-death', notify=lambda *args, **kwargs:None)
