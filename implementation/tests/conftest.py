import json
from pathlib import Path

import pytest

from medical_harness.core import Core
from medical_harness.store import Store
from thread_runner import ThreadRunner


@pytest.fixture
def snapshot():
    return json.loads((Path(__file__).parents[1] / "examples/synthetic-case.json").read_text())


@pytest.fixture
def core(tmp_path):
    value = Core(Store(tmp_path / "harness.sqlite3"), enable_fixtures=True, executor_runner=ThreadRunner())
    yield value
    value.close()


@pytest.fixture
def run(core, snapshot):
    case = core.create_case("alice", snapshot)
    return core.start_run("alice", case["case_id"], "eval", "run-1")


def allow_data(core, case, purpose='report'):
    from medical_harness.data_policy import POLICY_VERSION
    return core.data_permissions.update('alice', case['case_id'], {
        'snapshot_id': case['snapshot_id'], 'purpose': purpose, 'allowed': True, 'policy_version': POLICY_VERSION})
