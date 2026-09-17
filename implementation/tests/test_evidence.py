from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from medical_harness.api import create_app
from medical_harness.contracts import CaseEvidence, GateError, canonical
from medical_harness.render import TEMPLATE_VERSION


def cards(result):
    return {card['kind']: card for card in result['cards']}


def draft(core, run):
    job = core.create_draft_job('alice', run['id'])
    core.submit_proposal(job['job_id'], job['capability'], canonical({'sections':['forecast','policy','limitations']}))


def reviewed(core, run):
    core.execute_models('alice', run['id'])
    draft(core, run)
    for role in ['medical','ethics']:
        core.review('alice', run['id'], role)


def test_unstarted_case_reports_real_absence_without_starting_run(core, snapshot):
    case = core.create_case('alice', snapshot)
    result = core.case_evidence('alice', case['case_id'])
    CaseEvidence.model_validate(result)
    assert result['run_id'] is None and result['run_state'] == 'NOT_STARTED'
    assert result['release_allowed'] is False
    assert cards(result)['input']['status'] == 'verified'
    assert cards(result)['prediction']['reason'] == 'MODEL_NOT_CONFIGURED'
    assert cards(result)['policy']['reason'] == 'MODEL_NOT_CONFIGURED'
    with core.store.tx() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_missing_inputs_do_not_appear_ready(core, snapshot):
    snapshot['history'][2]['value'] = None
    snapshot['missing_mask'][2] = True
    case = core.create_case('alice', snapshot)
    result = core.case_evidence('alice', case['case_id'])
    assert result['missing_count'] == 1
    assert cards(result)['input']['reason'] == 'MISSING_INPUT'
    assert not result['release_allowed']


def test_evidence_projection_checks_binding_and_never_returns_payload(core, run):
    core.execute_models('alice', run['id'])
    result = core.case_evidence('alice', run['case_id'])
    assert all(cards(result)[kind]['status'] == 'verified' for kind in ['input','prediction','policy','safety'])
    assert cards(result)['prediction']['origin'] == 'fixture'
    assert not result['release_allowed']
    assert cards(result)['reviews']['reason'] == 'REVIEW_REQUIRED'
    text = canonical(result)
    for forbidden in ['action_value', 'values', 'payload', 'history', 'report_hash', 'capability']:
        assert forbidden not in text


def test_all_gates_only_pass_with_both_current_reviews_and_reads_do_not_mutate(core, run):
    core.execute_models('alice', run['id'])
    draft(core, run)
    core.review('alice', run['id'], 'medical')
    assert not core.case_evidence('alice', run['case_id'])['release_allowed']
    core.review('alice', run['id'], 'ethics')
    def ledger():
        with core.store.tx() as db:
            return {table: [tuple(r) for r in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                    for table in ['runs','jobs','artifacts','drafts','reviews','releases','events','outbox']}
    before = ledger()
    result = core.case_evidence('alice', run['case_id'])
    assert result['release_allowed'] and result['clinical_validation'] is False
    assert all(c['status'] == 'verified' for c in result['cards'])
    assert before == ledger()


@pytest.mark.parametrize('action,reason', [
    ('cancel','CANCELLED'), ('update','SNAPSHOT_CHANGED'), ('revoke','VERSION_REVOKED'), ('expire','RUN_EXPIRED')])
def test_terminal_or_changed_evidence_withdraws_all_model_authority(core, run, snapshot, action, reason):
    reviewed(core, run)
    assert core.case_evidence('alice', run['case_id'])['release_allowed']
    if action == 'cancel': core.cancel('alice', run['id'])
    elif action == 'update': core.update_case('alice', run['case_id'], snapshot)
    elif action == 'revoke': core.revoke_version(TEMPLATE_VERSION)
    else:
        with core.store.tx() as db:
            db.execute('UPDATE runs SET expires=0 WHERE id=?', (run['id'],))
    result = core.case_evidence('alice', run['case_id'])
    assert not result['release_allowed']
    assert cards(result)['prediction']['status'] == 'blocked'
    assert cards(result)['prediction']['reason'] == reason
    assert cards(result)['prediction']['digest'] is None
    assert cards(result)['reviews']['status'] == 'blocked'


def test_changed_draft_and_newer_run_do_not_reuse_old_success(core, run):
    reviewed(core, run)
    draft(core, run)
    assert not core.case_evidence('alice', run['case_id'])['release_allowed']
    for role in ['medical','ethics']: core.review('alice', run['id'], role)
    assert core.case_evidence('alice', run['case_id'])['release_allowed']
    newer = core.start_run('alice', run['case_id'], 'eval', 'newer')
    result = core.case_evidence('alice', run['case_id'])
    assert result['run_id'] == newer['id']
    assert cards(result)['prediction']['status'] == 'waiting'
    assert not result['release_allowed']


def test_corrupted_artifact_cannot_appear_verified(core, run):
    core.execute_models('alice', run['id'])
    with core.store.tx() as db:
        db.execute("UPDATE artifacts SET digest='broken' WHERE kind='prediction'")
    result = core.case_evidence('alice', run['case_id'])
    assert cards(result)['prediction']['reason'] == 'ARTIFACT_INTEGRITY'
    assert cards(result)['policy']['status'] == 'blocked'
    assert not result['release_allowed']


def test_failed_review_has_actual_blocking_reason(core, run):
    reviewed(core, run)
    draft(core, run)
    core._fixtures['ethics'] = replace(core._fixtures['ethics'], call=lambda r: {
        'verdict':'abstain','report_hash':r['report_hash'],'evidence_hash':r['evidence_hash'],'issues':[]})
    core.review('alice', run['id'], 'ethics')
    result = core.case_evidence('alice', run['case_id'])
    assert cards(result)['reviews']['reason'] == 'REVIEW_REJECTED'
    assert not result['release_allowed']


def test_unconfigured_research_run_and_api_owner_boundary(core, snapshot):
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'research', 'research')
    with pytest.raises(GateError, match='MODEL_NOT_CONFIGURED'):
        core.execute_models('alice', run['id'])
    with TestClient(create_app(core, {'a':'alice','b':'bob'})) as client:
        path = f"/cases/{case['case_id']}/evidence"
        assert client.get(path).status_code == 401
        assert client.get(path, headers={'Authorization':'Bearer b'}).status_code == 404
        assert client.post(path, headers={'Authorization':'Bearer a'}, json={'force':True}).status_code == 405
        assert client.get(path, headers={'Authorization':'Bearer a','Origin':'null'}).status_code == 403
        response = client.get(path, headers={'Authorization':'Bearer a'})
        assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
        assert cards(response.json())['prediction']['reason'] == 'MODEL_NOT_CONFIGURED'
        assert not response.json()['release_allowed']
