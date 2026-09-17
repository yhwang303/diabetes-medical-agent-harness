"""Unit-test-only fault injection for closures/events. Never imported by the service."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from threading import BoundedSemaphore

from medical_harness.contracts import GateError


class ThreadRunner:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.capacity = BoundedSemaphore(4)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def run(self, executor, request, *, timeout, **kwargs):
        if not self.capacity.acquire(blocking=False):
            raise GateError('EXECUTOR_BUSY', 503)
        future = self.pool.submit(executor.call, deepcopy(request))
        future.add_done_callback(lambda _: self.capacity.release())
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise GateError('EXECUTOR_TIMEOUT', 503) from exc
