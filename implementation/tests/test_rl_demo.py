import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest
from fastapi.testclient import TestClient

from medical_harness import desktop_client
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.store import Store
from thread_runner import ThreadRunner


def start(core, snapshot, scenario, key='demo'):
    case = core.create_case('alice', snapshot)
    return core.start_run('alice', case['case_id'], 'eval', key, case['snapshot_id'], scenario)


def ledger(core, run_id):
    with core.store.tx() as db:
        return {table: [dict(row) for row in db.execute(f'SELECT * FROM {table} WHERE run_id=?', (run_id,))]
                for table in ['jobs', 'artifacts', 'drafts', 'reviews', 'releases']}


@pytest.mark.parametrize('scenario,reason,policy_artifacts', [
    ('candidate', None, 1), ('abstain', 'POLICY_ABSTAIN', 0), ('unsupported', 'POLICY_UNSUPPORTED', 0),
    ('error', 'POLICY_ERROR', 0), ('invalid_output', 'INVALID_MODEL_OUTPUT', 0),
    ('parent_mismatch', 'FORECAST_PARENT_MISMATCH', 0), ('safety_rejected', 'ENGINEERING_SAFETY_REJECTED', 1)])
def test_real_core_gates_handle_each_fixed_scenario(core, snapshot, scenario, reason, policy_artifacts):
    run = start(core, snapshot, scenario)
    if reason and scenario != 'safety_rejected':
        with pytest.raises(GateError, match=reason):
            core.execute_models('alice', run['id'])
    else:
        core.execute_models('alice', run['id'])
    state = core.status('alice', run['id'])
    assert state['reason'] == reason and state['fixture_scenario'] == scenario
    assert state['state'] == ('SAFETY_ACCEPTED' if reason is None else 'BLOCKED')
    rows = ledger(core, run['id'])
    assert len(rows['jobs']) == 2 and len(rows['artifacts']) == 1 + policy_artifacts
    assert not rows['drafts'] and not rows['reviews'] and not rows['releases']
    assert [a['kind'] for a in rows['artifacts']].count('policy') == policy_artifacts
    evidence = core.case_evidence('alice', run['case_id'])
    assert not evidence['release_allowed'] and evidence['fixture_scenario'] == scenario
    assert evidence['cards'][2]['status'] == ('verified' if reason is None else 'blocked')
    with pytest.raises(GateError):
        core.release('alice', run['id'])
    text = canonical(evidence) + canonical(state) + canonical(core.events('alice', run['id']))
    for forbidden in ['action_value', 'action_unit', 'values', 'payload', 'capability']:
        assert forbidden not in text
    if reason is None:
        body = json.loads(next(a['body'] for a in rows['artifacts'] if a['kind'] == 'policy'))
        assert body['producer'] == 'fixture.policy.candidate' and body['origin'] == 'fixture'
        prediction = next(a for a in rows['artifacts'] if a['kind'] == 'prediction')
        assert body['parent_hash'] == prediction['digest']


def test_timeout_really_waits_and_discards_late_result(tmp_path, snapshot):
    core = Core(Store(tmp_path / 'timeout.sqlite3'), enable_fixtures=True, executor_timeout=0.025,
                executor_runner=ThreadRunner())
    finished = Event()
    executor = core._policy_scenarios['timeout']
    def delayed(request):
        try:
            return executor.call(request)
        finally:
            finished.set()
    core._policy_scenarios['timeout'] = replace(executor, call=delayed)
    try:
        run = start(core, snapshot, 'timeout')
        with pytest.raises(GateError, match='EXECUTOR_TIMEOUT'):
            core.execute_models('alice', run['id'])
        assert not finished.is_set()
        assert finished.wait(1)
        assert core.status('alice', run['id'])['state'] == 'UNAVAILABLE'
        rows = ledger(core, run['id'])
        assert len(rows['artifacts']) == 1 and rows['artifacts'][0]['kind'] == 'prediction'
        assert [j for j in rows['jobs'] if j['step'] == 'policy'][0]['status'] == 'FAILED'
        assert core.case_evidence('alice', run['case_id'])['cards'][2]['reason'] == 'EXECUTOR_TIMEOUT'
        assert not rows['releases']
    finally:
        core.close()


def test_scenarios_are_isolated_between_concurrent_runs_and_normal_fixture(core, snapshot):
    good = start(core, snapshot, 'candidate', 'good')
    bad = start(core, snapshot, 'abstain', 'bad')
    baseline = start(core, snapshot, None, 'baseline')
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(core.execute_models, 'alice', good['id'])
        two = pool.submit(core.execute_models, 'alice', bad['id'])
        assert one.result()['state'] == 'SAFETY_ACCEPTED'
        with pytest.raises(GateError, match='POLICY_ABSTAIN'):
            two.result()
    assert core.execute_models('alice', baseline['id'])['state'] == 'SAFETY_ACCEPTED'
    body = json.loads(next(a['body'] for a in ledger(core, baseline['id'])['artifacts'] if a['kind'] == 'policy'))
    assert body['producer'] == 'fixture.policy'


def test_scenario_is_frozen_in_idempotency_identity_and_restart(core, snapshot):
    run = start(core, snapshot, 'parent_mismatch')
    with pytest.raises(GateError, match='IDEMPOTENCY_CONFLICT'):
        core.start_run('alice', run['case_id'], 'eval', 'demo', run['snapshot_id'], 'candidate')
    restarted = Core(Store(core.store.path), enable_fixtures=True)
    try:
        same = restarted.start_run('alice', run['case_id'], 'eval', 'demo', run['snapshot_id'], 'parent_mismatch')
        assert same == run
        with pytest.raises(GateError, match='FORECAST_PARENT_MISMATCH'):
            restarted.execute_models('alice', run['id'])
    finally:
        restarted.close()


def test_additive_migration_preserves_pre_scenario_data_and_idempotency(core, run):
    core.execute_models('alice', run['id'])
    before = ledger(core, run['id'])
    with sqlite3.connect(core.store.path) as db:
        db.execute('ALTER TABLE runs DROP COLUMN fixture_scenario')
    restarted = Core(Store(core.store.path), enable_fixtures=True)
    try:
        same = restarted.start_run('alice', run['case_id'], 'eval', 'run-1')
        assert same['id'] == run['id'] and same['fixture_scenario'] is None
        assert ledger(restarted, run['id']) == before
        assert restarted.case_evidence('alice', run['case_id'])['cards'][2]['status'] == 'verified'
    finally:
        restarted.close()


@pytest.mark.parametrize('mode,source,enabled,reason', [
    ('research','synthetic',True,'FIXTURE_FORBIDDEN'), ('eval','historical',True,'FIXTURE_REQUIRES_SYNTHETIC_CASE'),
    ('eval','simulation',True,'FIXTURE_REQUIRES_SYNTHETIC_CASE'), ('eval','synthetic',False,'FIXTURES_DISABLED')])
def test_no_fixture_escape_into_research_or_non_synthetic(core, snapshot, mode, source, enabled, reason):
    snapshot['source'] = source
    case = core.create_case('alice', snapshot)
    core.enable_fixtures = enabled
    with pytest.raises(GateError, match=reason):
        core.start_run('alice', case['case_id'], mode, 'bad', case['snapshot_id'], 'candidate')
    with core.store.tx() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_missing_input_and_revoked_scenario_never_execute(core, snapshot):
    snapshot['history'][0]['value'] = None
    snapshot['missing_mask'][0] = True
    run = start(core, snapshot, 'candidate')
    assert core.execute_models('alice', run['id'])['reason'] == 'MISSING_INPUT'
    assert not ledger(core, run['id'])['jobs']
    core.revoke_version(core._policy_scenarios['candidate'].version)
    with pytest.raises(GateError, match='VERSION_REVOKED'):
        start(core, snapshot, 'candidate', 'revoked')


def test_scenario_denial_audit_failure_rolls_back_run_creation(core, snapshot, monkeypatch):
    case = core.create_case('alice', snapshot)
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('audit unavailable')
    monkeypatch.setattr(core, '_event', fail)
    with pytest.raises(sqlite3.Error):
        core.start_run('alice', case['case_id'], 'eval', 'demo', case['snapshot_id'], 'candidate')
    with core.store.tx() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_bridge_uses_only_fixed_scenario_and_core_result(core, snapshot, monkeypatch):
    case = core.create_case('alice', snapshot)
    payload = {'case_id':case['case_id'], 'snapshot_id':case['snapshot_id'], 'idempotency_key':'a'*32, 'scenario':'abstain'}
    with TestClient(create_app(core, {'a':'alice'})) as client:
        def request(port, method, path, body):
            response = client.request(method, path, content=body, headers={'Authorization':'Bearer a'})
            if response.status_code >= 400:
                raise GateError(response.json()['error'])
            return response.json()
        monkeypatch.setattr(desktop_client, '_request', request)
        for _ in range(2):
            with pytest.raises(GateError, match='POLICY_ABSTAIN'):
                desktop_client.call(1234, 'rl-demo', payload)
        with core.store.tx() as db:
            assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1
            assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 2
        assert core.case_evidence('alice', case['case_id'])['fixture_scenario'] == 'abstain'
        for extra in [{'scenario':'custom'}, {'action_value':0.5}, {'force':True}, {'mode':'research'}]:
            response = client.post('/runs', json={'case_id':case['case_id'], 'mode':'eval', 'idempotency_key':'forged', 'fixture_scenario':'abstain', **extra}, headers={'Authorization':'Bearer a'})
            assert response.status_code in (403,422)


@pytest.mark.parametrize('patch', [{'scenario':'../execute'}, {'scenario':{}}, {'mode':'research'}, {'action_value':0.1}])
def test_bridge_rejects_custom_scenario_or_model_fields_before_http(monkeypatch, patch):
    def unexpected(*args):
        pytest.fail('Invalid bridge request reached transport')
    monkeypatch.setattr(desktop_client, '_request', unexpected)
    with pytest.raises(GateError, match='DESKTOP_OPERATION_FORBIDDEN'):
        desktop_client.call(1234, 'rl-demo', {'case_id':'a'*32, 'snapshot_id':'b'*32, 'idempotency_key':'c'*32, 'scenario':'candidate', **patch})
