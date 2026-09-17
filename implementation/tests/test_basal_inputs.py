"""P01a facts and execution boundaries; no real patient data or provider calls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from medical_harness.agent_tasks import AgentTasks
from medical_harness.api import create_app
from medical_harness.basal_inputs import BasalResearchInput, parse_snapshot
from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core


@pytest.fixture
def basal():
    return json.loads((Path(__file__).parents[1] / 'examples/basal-research-case.json').read_text())


@pytest.mark.parametrize('source', ['synthetic', 'simulation'])
def test_research_input_roundtrip_and_restart(core, basal, source):
    basal['source'] = source
    for provenance in (basal['provenance'], basal['treatment']['provenance'], basal['insulin_history']['provenance']):
        provenance['kind'] = 'synthetic' if source == 'synthetic' else 'simulator'
    raw = {'file_name': 'research.json', 'content': canonical(basal)}
    case = core.import_case('alice', raw)
    assert core.import_case('alice', raw)['duplicate']
    timeline = core.input_timeline('alice', case['case_id'])
    assert timeline['snapshot'] == basal
    assert timeline['snapshot_hash'] == digest(basal)
    restarted = Core(core.store)
    try:
        assert restarted.input_timeline('alice', case['case_id']) == timeline
    finally:
        restarted.close()


@pytest.mark.parametrize('path,value', [
    (('contract_version',), 'unknown-v2'), (('contract_version',), None),
    (('use_profile',), 'clinical'), (('cohort',), 'adult_t2d'), (('source',), 'historical'),
    (('unit',), 'mmol/L'), (('treatment', 'insulin'), 'real_drug'),
    (('treatment', 'route'), 'injection'), (('treatment', 'regimen'), 'basal_bolus'),
    (('provenance', 'source_ref'), 'different'),
    (('treatment', 'provenance', 'subject_ref'), 'other-subject'),
    (('insulin_history', 'provenance', 'kind'), 'simulator'),
    (('insulin_history', 'provenance', 'subject_ref'), 'other-subject'),
    (('provenance', 'available_at'), '2026-09-09T10:01:00+08:00'),
    (('provenance', 'available_at'), '2026-09-09T09:59:00+08:00'),
    (('treatment', 'provenance', 'available_at'), '2026-09-09T10:01:00+08:00'),
    (('provenance', 'available_at'), '2026-09-09T10:00:00'),
    (('provenance', 'locator'), ''),
    (('insulin_history', 'deliveries', 0, 'record_kind'), 'recommended'),
    (('insulin_history', 'deliveries', 0, 'record_kind'), 'accepted'),
    (('insulin_history', 'deliveries', 0, 'unit'), 'U/h'),
    (('insulin_history', 'deliveries', 0, 'delivered_units'), -1),
    (('insulin_history', 'deliveries', 0, 'delivered_units'), True),
    (('insulin_history', 'deliveries', 0, 'delivered_units'), '0.1'),
    (('insulin_history', 'deliveries', 0, 'delivered_units'), float('nan')),
    (('insulin_history', 'deliveries', 0, 'end_time'), '2026-09-09T09:35:00+08:00'),
    (('insulin_history', 'deliveries', 0, 'start_time'), '2026-09-09T09:36:00+08:00'),
    (('insulin_history', 'coverage_end'), '2026-09-09T09:59:00+08:00'),
    (('insulin_history', 'provenance', 'available_at'), '2026-09-09T09:59:00+08:00'),
    (('insulin_history', 'deliveries'), []),
])
def test_invalid_facts_cannot_be_frozen(core, basal, path, value):
    target = basal
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        core.create_case('alice', basal)
    with core.store.tx() as db:
        assert db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0] == 0


@pytest.mark.parametrize('fault', ['duplicate', 'overlap', 'gap', 'future'])
def test_delivery_intervals_cannot_hide_double_counting_or_future(core, basal, fault):
    history = basal['insulin_history']
    first = history['deliveries'][0]
    first['end_time'] = '2026-09-09T09:50:00+08:00'
    second = dict(first, record_id='second', start_time=first['end_time'], end_time=basal['decision_time'])
    history['deliveries'].append(second)
    if fault == 'duplicate': second['record_id'] = first['record_id']
    elif fault == 'overlap': second['start_time'] = '2026-09-09T09:49:00+08:00'
    elif fault == 'gap': second['start_time'] = '2026-09-09T09:51:00+08:00'
    else:
        second['end_time'] = history['coverage_end'] = history['provenance']['available_at'] = '2026-09-09T10:05:00+08:00'
    with pytest.raises(ValidationError):
        core.create_case('alice', basal)


@pytest.mark.parametrize('field,reason', [('treatment', 'TREATMENT_REQUIRED'),
    ('insulin_history', 'INSULIN_HISTORY_REQUIRED'), ('cgm', 'MISSING_INPUT'),
    ('complete', 'INPUT_PROFILE_NOT_CONFIGURED')])
def test_unknown_stays_unknown_and_no_execution_path(core, basal, field, reason):
    if field == 'cgm':
        basal['history'][0]['value'] = None
        basal['missing_mask'][0] = True
    elif field != 'complete': basal[field] = None
    case = core.create_case('alice', basal)
    assert core.input_timeline('alice', case['case_id'])['snapshot'] == basal
    evidence = core.case_evidence('alice', case['case_id'])
    assert not evidence['release_allowed']
    assert evidence['cards'][0]['reason'] == reason
    for mode in ('eval', 'research'):
        with pytest.raises(GateError, match=reason):
            core.start_run('alice', case['case_id'], mode, mode)
    class NoReporter:
        runner = SimpleNamespace(close=lambda: None)
        def generate(self, *args):
            pytest.fail('new input must never reach report or model execution')
        @property
        def key_path(self):
            pytest.fail('must reject before accessing provider credentials')
    tasks = AgentTasks(core, NoReporter())
    try:
        with pytest.raises(GateError, match=reason):
            tasks.start('alice', dict(case, idempotency_key=uuid4().hex))
    finally:
        tasks.close()
    with core.store.tx() as db:
        for table in ('runs', 'jobs', 'artifacts', 'drafts', 'reviews', 'releases', 'agent_tasks'):
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_zero_is_recorded_delivery_not_unknown_and_no_implicit_facts(core, basal):
    basal['insulin_history']['deliveries'][0]['delivered_units'] = 0.0
    assert BasalResearchInput.model_validate(basal).insulin_history.deliveries[0].delivered_units == 0.0
    for field in ('treatment', 'insulin_history', 'contract_version'):
        incomplete = deepcopy(basal)
        del incomplete[field]
        with pytest.raises(ValidationError):
            parse_snapshot(incomplete)


def test_contiguous_deliveries_and_equivalent_offsets_are_accepted(basal):
    first = basal['insulin_history']['deliveries'][0]
    first['end_time'] = '2026-09-09T09:50:00+08:00'
    basal['insulin_history']['deliveries'].append(dict(first, record_id='second',
        start_time='2026-09-09T01:50:00+00:00', end_time='2026-09-09T02:00:00+00:00'))
    assert parse_snapshot(basal).model_dump() == basal


def test_execution_rechecks_profile_even_for_preexisting_run(core, run, basal):
    # A recomputed storage hash cannot make a new input compatible with a legacy job.
    with core.store.tx() as db:
        db.execute('UPDATE snapshots SET body=?,digest=? WHERE id=?',
                   (canonical(basal), digest(basal), run['snapshot_id']))
    for action in (core.execute_prediction, core.execute_models, core.create_draft_job, core.release):
        with pytest.raises(GateError, match='INPUT_PROFILE_NOT_CONFIGURED'):
            action('alice', run['id'])
    with core.store.tx() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_new_fields_cannot_enter_legacy_fixture_or_become_authority(core, snapshot, basal):
    for field, value in [('insulin_history', basal['insulin_history']), ('use_profile', basal['use_profile']),
                         ('verified', True), ('origin', 'model')]:
        with pytest.raises(ValidationError):
            core.create_case('alice', dict(snapshot, **{field: value}))
    for field in ('prediction', 'recommended_action', 'verified', 'release_id'):
        with pytest.raises(ValidationError):
            core.create_case('alice', dict(basal, **{field: 'untrusted'}))


def test_http_scope_denial_audit_and_snapshot_replacement(core, run, basal):
    # An old accepted fixture report cannot survive replacement with research facts.
    core.execute_models('alice', run['id'])
    job = core.create_draft_job('alice', run['id'])
    core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
    for role in ('medical', 'ethics'): core.review('alice', run['id'], role)
    release = core.release('alice', run['id'])
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        auth = {'Authorization': 'Bearer a'}
        assert client.put(f"/cases/{run['case_id']}/snapshot", headers=auth, json=basal).status_code == 200
        assert client.get(f"/releases/{release['release_id']}", headers=auth).json()['error'] == 'SNAPSHOT_CHANGED'
        response = client.post('/runs', headers=auth,
            json={'case_id': run['case_id'], 'mode': 'eval', 'idempotency_key': 'blocked'})
        assert response.status_code == 409 and response.json() == {'error': 'INPUT_PROFILE_NOT_CONFIGURED'}
        assert client.get(f"/cases/{run['case_id']}/timeline", headers={'Authorization': 'Bearer b'}).status_code == 404
    assert core.status('alice', run['id'])['state'] == 'INVALIDATED'
    with core.store.tx() as db:
        event = db.execute("SELECT body FROM events WHERE kind='request_denied' ORDER BY sequence DESC").fetchone()
        assert 'subject_ref' not in event['body'] and 'delivered_units' not in event['body']


def test_legacy_snapshot_and_released_report_bytes_are_preserved(core, run, snapshot):
    core.execute_models('alice', run['id'])
    job = core.create_draft_job('alice', run['id'])
    core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
    for role in ('medical', 'ethics'): core.review('alice', run['id'], role)
    release = core.release('alice', run['id'])
    with core.store.tx() as db:
        before = {table: [tuple(r) for r in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                  for table in ('snapshots', 'runs', 'jobs', 'artifacts', 'drafts', 'reviews', 'releases')}
    assert core.input_timeline('alice', run['case_id'])['snapshot'] == snapshot
    assert parse_snapshot(snapshot).model_dump() == snapshot
    restarted = Core(core.store, enable_fixtures=True)
    try:
        assert restarted.get_release('alice', release['release_id']) == core.get_release('alice', release['release_id'])
    finally:
        restarted.close()
    with core.store.tx() as db:
        after = {table: [tuple(r) for r in db.execute(f'SELECT * FROM {table} ORDER BY rowid')] for table in before}
    assert before == after
