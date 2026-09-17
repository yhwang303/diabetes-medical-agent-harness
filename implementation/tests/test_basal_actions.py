"""P01b synthetic contract chain, never a treatment or model-performance evaluation."""

from dataclasses import replace
from pathlib import Path
from threading import Event
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

from fastapi.testclient import TestClient
import pytest

from medical_harness.adapters import basal_rate_fixture_policy as basal_fixture_policy
from medical_harness.api import create_app
from medical_harness.basal_actions import PROFILE_VERSION
from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core
from medical_harness.store import Store
from thread_runner import ThreadRunner


@pytest.fixture
def basal_case():
    return json.loads((Path(__file__).parents[1] / 'examples/basal-research-case.json').read_text())


@pytest.fixture
def setup(tmp_path, basal_case):
    core = Core(Store(tmp_path / 'basal.sqlite3'), enable_fixtures=True, enable_basal_fixtures=True,
                executor_runner=ThreadRunner(), clock=lambda: 1000)
    case = core.create_case('alice', basal_case)
    run = core.start_run('alice', case['case_id'], 'eval', 'basal')
    yield core, run
    core.close()


def draft(core, run):
    job = core.create_draft_job('alice', run['id'])
    return core.submit_proposal(job['job_id'], job['capability'],
                               canonical({'sections': ['forecast', 'policy', 'limitations']}))


def release(core, run):
    core.execute_models('alice', run['id'])
    draft(core, run)
    for role in ('medical', 'ethics'): core.review('alice', run['id'], role)
    return core.release('alice', run['id'])


def count(core, table):
    with core.store.tx() as db:
        return db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]


def test_full_chain_bound_profile_and_actual_history_unchanged(setup, basal_case):
    core, run = setup
    released = release(core, run)
    result = core.get_release('alice', released['release_id'])
    report = result['report']
    assert report['clinical_use'] is False and report['origin'] == 'fixture'
    assert report['valid_until'] == 1300 and report['time_basis'] == 'synthetic_replay'
    action = report['sections'][1]['action_display']
    assert (action['rate_u_per_min'], action['rate_u_per_hour'], action['interval_total_units']) == ('0.01', '0.6', '0.05')
    assert action['duration_minutes'] == 5 and not action['action_executed']
    assert '不是单次注射' in action['notice']
    assert core.input_timeline('alice', run['case_id'])['snapshot'] == basal_case
    with core.store.tx() as db:
        row = db.execute('SELECT * FROM run_profiles').fetchone()
        profile = json.loads(row['body'])
        assert digest(profile) == row['digest']
        assert profile['binding']['snapshot_hash'] == digest(basal_case)
        prediction, phash = core._artifact(db, core._run(db, run['id'], 'alice'), 'prediction')
        policy, _ = core._artifact(db, core._run(db, run['id'], 'alice'), 'policy')
        assert prediction['payload']['profile_hash'] == policy['payload']['profile_hash'] == row['digest']
        assert policy['payload']['forecast_parent_hash'] == phash
    assert count(core, 'reviews') == 2 and count(core, 'releases') == 1
    assert core.case_evidence('alice', run['case_id'])['release_allowed']
    # All model/review callbacks completed through Core, with no implicit execution feedback.
    assert 'delivered_units' not in canonical(core.events('alice', run['id']))


@pytest.mark.parametrize('rate,hour,total', [(0.0, '0', '0'), (0.000001, '0.00006', '0.000005'),
                                          (0.05, '3', '0.25')])
def test_exact_conversion_including_valid_zero(setup, rate, hour, total):
    core, run = setup
    def call(request):
        output = basal_fixture_policy(request)
        output['action'].update(action_value=rate, action_unit='U/min')
        return output
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)
    released = release(core, run)
    action = core.get_release('alice', released['release_id'])['report']['sections'][1]['action_display']
    assert action['rate_u_per_hour'] == hour and action['interval_total_units'] == total


@pytest.mark.parametrize('path,value,reason', [
    (('action', 'action_unit'), 'mg/h', 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_kind'), 'bolus', 'INVALID_MODEL_OUTPUT'),
    (('action', 'route'), 'injection', 'INVALID_MODEL_OUTPUT'),
    (('action', 'insulin'), 'real_drug', 'INVALID_MODEL_OUTPUT'),
    (('action', 'duration_minutes'), 10, 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_unit'), 'U', 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_value'), -1, 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_value'), float('inf'), 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_value'), True, 'INVALID_MODEL_OUTPUT'),
    (('action', 'action_value'), '0.01', 'INVALID_MODEL_OUTPUT'),
    (('action', 'end_time'), '2026-09-09T10:06:00+08:00', 'INVALID_MODEL_OUTPUT'),
    (('action', 'decision_time'), '2026-09-09T10:00:00', 'INVALID_MODEL_OUTPUT'),
    (('action', 'interval_total_units'), 999, 'INVALID_MODEL_OUTPUT'),
    (('profile_hash',), 'forged', 'MODEL_PROFILE_BINDING'),
    (('forecast_parent_hash',), 'another-prediction', 'FORECAST_PARENT_MISMATCH'),
    (('input_digest',), 'forged', 'INPUT_DIGEST_MISMATCH'),
    (('verified',), True, 'INVALID_MODEL_OUTPUT'),
])
def test_invalid_policy_never_registers(setup, path, value, reason):
    core, run = setup
    def call(request):
        output = basal_fixture_policy(request)
        target = output
        for key in path[:-1]: target = target[key]
        target[path[-1]] = value
        return output
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)
    with pytest.raises(GateError, match=reason): core.execute_models('alice', run['id'])
    assert count(core, 'artifacts') == 1
    with pytest.raises(GateError): draft(core, run)
    assert count(core, 'releases') == 0


@pytest.mark.parametrize('status,reason', [('abstain', 'insufficient_input'), ('unsupported', 'unsupported'), ('error', 'model_error')])
def test_abstention_cannot_substitute_zero(setup, status, reason):
    core, run = setup
    def call(request):
        return dict(basal_fixture_policy(request), status=status, reason=reason, action=None)
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)
    with pytest.raises(GateError, match='POLICY_' + status.upper()): core.execute_models('alice', run['id'])
    assert count(core, 'artifacts') == 1 and count(core, 'drafts') == 0


def test_wrong_decision_interval_and_safety_bound(setup):
    core, run = setup
    def call(request):
        output = basal_fixture_policy(request)
        output['action'].update(decision_time='2026-09-09T10:05:00+08:00', end_time='2026-09-09T10:10:00+08:00')
        return output
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)
    with pytest.raises(GateError, match='ACTION_TIME_BINDING'): core.execute_models('alice', run['id'])
    assert count(core, 'artifacts') == 1


def test_engineering_limit_rejects_without_replacement(setup):
    core, run = setup
    def call(request):
        output = basal_fixture_policy(request)
        output['action'].update(action_value=0.1, action_unit='U/min')
        return output
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)
    assert core.execute_models('alice', run['id'])['reason'] == 'ENGINEERING_SAFETY_REJECTED'
    assert count(core, 'drafts') == count(core, 'releases') == 0


@pytest.mark.parametrize('fault,reason', [('config', 'INPUT_PROFILE_CHANGED'), ('digest', 'INPUT_PROFILE_INTEGRITY'),
    ('missing', 'INPUT_PROFILE_MISSING'), ('expiry', 'RUN_EXPIRED'), ('extend_expiry', 'INPUT_PROFILE_CHANGED'),
    ('cancel', 'CANCELLED'), ('snapshot', 'SNAPSHOT_CHANGED'), ('revocation', 'VERSION_REVOKED'),
    ('executor', 'EXECUTOR_PROFILE_MISMATCH'), ('disabled', 'INPUT_PROFILE_NOT_CONFIGURED')])
def test_release_reads_recheck_every_dependency(setup, basal_case, monkeypatch, fault, reason):
    core, run = setup
    released = release(core, run)
    if fault == 'config':
        import medical_harness.core as module
        original = module.fixture_profile
        monkeypatch.setattr(module, 'fixture_profile', lambda: dict(original(), max_wall_seconds=301))
    elif fault in ('digest', 'missing', 'extend_expiry'):
        with core.store.tx() as db:
            db.execute({'digest': "UPDATE run_profiles SET digest='wrong'", 'missing': 'DELETE FROM run_profiles',
                        'extend_expiry': 'UPDATE runs SET expires=9999'}[fault])
    elif fault == 'expiry': core.clock = lambda: 1300
    elif fault == 'cancel': core.cancel('alice', run['id'])
    elif fault == 'snapshot': core.update_case('alice', run['case_id'], basal_case)
    elif fault == 'revocation': core.revoke_version(PROFILE_VERSION)
    elif fault == 'disabled': core.enable_basal_fixtures = False
    else: core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], identity='forged')
    with pytest.raises(GateError, match=reason): core.get_release('alice', released['release_id'])
    with pytest.raises(GateError): core.release('alice', run['id'])
    assert not core.case_evidence('alice', run['case_id'])['release_allowed']


def test_profile_from_other_run_cannot_be_replayed(setup):
    core, run = setup
    another = core.start_run('alice', run['case_id'], 'eval', 'another')
    with core.store.tx() as db:
        row = db.execute('SELECT body,digest FROM run_profiles WHERE run_id=?', (run['id'],)).fetchone()
        db.execute('UPDATE run_profiles SET body=?,digest=? WHERE run_id=?', (*row, another['id']))
    with pytest.raises(GateError, match='INPUT_PROFILE_CHANGED'): core.execute_models('alice', another['id'])
    assert count(core, 'artifacts') == 0


@pytest.mark.parametrize('fault,reason', [('research', 'INPUT_PROFILE_NOT_CONFIGURED'),
    ('source', 'FIXTURE_REQUIRES_SYNTHETIC_CASE'), ('treatment', 'TREATMENT_REQUIRED'),
    ('insulin_history', 'INSULIN_HISTORY_REQUIRED'), ('window', 'INSUFFICIENT_INPUT_WINDOW'),
    ('scenario', 'FIXTURE_SCENARIO_PROFILE_MISMATCH')])
def test_profile_preflight_no_jobs(setup, basal_case, fault, reason):
    core, _ = setup
    if fault == 'source':
        basal_case['source'] = 'simulation'
        for provenance in [basal_case['provenance'], basal_case['treatment']['provenance'], basal_case['insulin_history']['provenance']]:
            provenance['kind'] = 'simulator'
    elif fault in ('treatment', 'insulin_history'): basal_case[fault] = None
    elif fault == 'window':
        basal_case['insulin_history']['coverage_start'] = '2026-09-09T09:55:00+08:00'
        basal_case['insulin_history']['deliveries'][0]['start_time'] = '2026-09-09T09:55:00+08:00'
    case = core.create_case('alice', basal_case)
    with pytest.raises(GateError, match=reason):
        core.start_run('alice', case['case_id'], 'research' if fault == 'research' else 'eval', 'new',
                       fixture_scenario='candidate' if fault == 'scenario' else None)
    assert count(core, 'jobs') == 0


def test_expiry_and_cancel_between_model_return_and_registration(setup):
    core, run = setup
    def late(request):
        core.clock = lambda: 1300
        return basal_fixture_policy(request)
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=late)
    with pytest.raises(GateError, match='RUN_EXPIRED'): core.execute_models('alice', run['id'])
    assert count(core, 'artifacts') == 1


def test_cancel_running_policy_discards_late_output(setup):
    core, run = setup
    entered, proceed = Event(), Event()
    def delayed(request):
        entered.set()
        assert proceed.wait(3)
        return basal_fixture_policy(request)
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=delayed)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(core.execute_models, 'alice', run['id'])
        assert entered.wait(3)
        core.cancel('alice', run['id'])
        proceed.set()
        with pytest.raises(GateError, match='CANCELLED'): future.result()
    assert count(core, 'artifacts') == 1 and count(core, 'releases') == 0


def test_review_and_audit_gates_still_apply(setup, monkeypatch):
    core, run = setup
    core.execute_models('alice', run['id'])
    draft(core, run)
    core.review('alice', run['id'], 'medical')
    with pytest.raises(GateError, match='REVIEW_REQUIRED'): core.release('alice', run['id'])
    core.review('alice', run['id'], 'ethics')
    draft(core, run)
    with pytest.raises(GateError, match='REVIEW_REQUIRED'): core.release('alice', run['id'])
    for role in ('medical', 'ethics'): core.review('alice', run['id'], role)
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == 'release_committed': raise sqlite3.OperationalError('audit unavailable')
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    with pytest.raises(sqlite3.OperationalError): core.release('alice', run['id'])
    assert count(core, 'releases') == 0


def test_http_authorization_and_no_client_configuration(setup):
    core, run = setup
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        auth = {'Authorization': 'Bearer a'}
        assert client.post('/runs', headers=auth, json={'case_id': run['case_id'], 'mode': 'eval',
            'idempotency_key': 'attack', 'profile': {'clinical_use': True}}).status_code == 422
        assert client.post(f"/runs/{run['id']}/execute", headers={'Authorization': 'Bearer b'}).status_code == 404
        assert client.post(f"/runs/{run['id']}/execute", headers=auth, json={'action_value': 99}).status_code == 422
        assert client.post(f"/runs/{run['id']}/execute", headers=auth).status_code == 200
        status = client.get(f"/runs/{run['id']}", headers=auth).json()
        assert 'action' not in status and 'profile_hash' not in status
    assert count(core, 'drafts') == 0


def test_modified_rendered_total_invalidates_even_if_hash_recomputed(setup):
    core, run = setup
    release(core, run)
    with core.store.tx() as db:
        row = db.execute('SELECT * FROM drafts').fetchone()
        body = json.loads(row['body'])
        body['sections'][1]['action_display']['interval_total_units'] = '999'
        db.execute('UPDATE drafts SET body=?,digest=?', (canonical(body), digest(body)))
    with pytest.raises(GateError, match='TEMPLATE_MISMATCH'): core.release('alice', run['id'])


def test_real_worker_registration_and_restart(tmp_path, basal_case):
    core = Core(Store(tmp_path / 'process.sqlite3'), enable_fixtures=True, enable_basal_fixtures=True)
    try:
        case = core.create_case('alice', basal_case)
        run = core.start_run('alice', case['case_id'], 'eval', 'real-process')
        released = release(core, run)
        before = core.get_release('alice', released['release_id'])
        stops = [event['body'] for event in core.events('alice', run['id']) if event['kind'] == 'worker_stopped']
        assert len(stops) == 4 and all(event['exit_code'] == 0 for event in stops)
    finally: core.close()
    restarted = Core(core.store, enable_fixtures=True, enable_basal_fixtures=True)
    try:
        assert restarted.get_release('alice', released['release_id']) == before
    finally: restarted.close()
