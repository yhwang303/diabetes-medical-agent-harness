from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from medical_harness import desktop_client
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical


def counts(core):
    with core.store.tx() as db:
        return {t: db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                for t in ['runs', 'jobs', 'artifacts', 'drafts', 'reviews', 'releases']}


@pytest.fixture
def bridge(core, monkeypatch):
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        def request(port, method, path, body):
            response = client.request(method, path, content=body, headers={'Authorization': 'Bearer a'})
            if response.status_code >= 400:
                raise GateError(response.json()['error'])
            return response.json()
        monkeypatch.setattr(desktop_client, '_request', request)
        yield client


def payload(case):
    return {'case_id': case['case_id'], 'snapshot_id': case['snapshot_id'], 'idempotency_key': 'a' * 32}


def test_desktop_prediction_only_idempotent_and_no_result_leak(core, snapshot, bridge):
    case = core.create_case('alice', snapshot)
    def unexpected(_):
        pytest.fail('Prediction-only operation must never call another executor')
    for kind in ['policy', 'medical', 'ethics']:
        core._fixtures[kind] = replace(core._fixtures[kind], call=unexpected)
    result = desktop_client.call(1234, 'prediction', payload(case))
    assert result['state'] == 'PREDICTION_ACCEPTED' and result['currently_valid']
    assert result == desktop_client.call(1234, 'prediction', payload(case))
    assert counts(core) == dict(runs=1, jobs=1, artifacts=1, drafts=0, reviews=0, releases=0)
    evidence = core.case_evidence('alice', case['case_id'])
    assert evidence['cards'][1]['origin'] == 'fixture'
    assert evidence['cards'][2]['reason'] == 'MISSING_POLICY'
    assert not evidence['release_allowed']
    for forbidden in ['values', 'payload', 'action_value', 'capability']:
        assert forbidden not in canonical(result) + canonical(evidence)


def test_lost_prediction_response_can_retry_same_request_without_duplicate(core, snapshot, bridge, monkeypatch):
    case = core.create_case('alice', snapshot)
    request = desktop_client._request
    def lost_ack(port, method, path, body):
        result = request(port, method, path, body)
        if path.endswith('/prediction'):
            raise OSError('lost response after commit')
        return result
    monkeypatch.setattr(desktop_client, '_request', lost_ack)
    with pytest.raises(OSError):
        desktop_client.call(1234, 'prediction', payload(case))
    monkeypatch.setattr(desktop_client, '_request', request)
    assert desktop_client.call(1234, 'prediction', payload(case))['state'] == 'PREDICTION_ACCEPTED'
    assert counts(core)['jobs'] == counts(core)['artifacts'] == counts(core)['runs'] == 1


@pytest.mark.parametrize('source', ['historical', 'simulation'])
def test_desktop_fixture_rejects_non_synthetic_before_run(core, snapshot, bridge, source):
    snapshot['source'] = source
    case = core.create_case('alice', snapshot)
    with pytest.raises(GateError, match='FIXTURE_REQUIRES_SYNTHETIC_CASE'):
        desktop_client.call(1234, 'prediction', payload(case))
    assert not any(counts(core).values())


def test_desktop_respects_service_fixture_opt_in(core, snapshot, bridge):
    core.enable_fixtures = False
    case = core.create_case('alice', snapshot)
    with pytest.raises(GateError, match='FIXTURES_DISABLED'):
        desktop_client.call(1234, 'prediction', payload(case))
    assert not any(counts(core).values())


def test_stale_desktop_snapshot_cannot_create_run(core, snapshot, bridge):
    case = core.create_case('alice', snapshot)
    core.update_case('alice', case['case_id'], snapshot)
    with pytest.raises(GateError, match='SNAPSHOT_CHANGED'):
        desktop_client.call(1234, 'prediction', payload(case))
    assert not any(counts(core).values())


def test_snapshot_change_between_two_bridge_requests_blocks_execution(core, snapshot, bridge, monkeypatch):
    case = core.create_case('alice', snapshot)
    request = desktop_client._request
    def changed(port, method, path, body):
        result = request(port, method, path, body)
        if path == '/runs':
            core.update_case('alice', case['case_id'], snapshot)
        return result
    monkeypatch.setattr(desktop_client, '_request', changed)
    with pytest.raises(GateError, match='SNAPSHOT_CHANGED'):
        desktop_client.call(1234, 'prediction', payload(case))
    assert counts(core)['runs'] == 1 and counts(core)['jobs'] == counts(core)['artifacts'] == 0


def test_missing_input_cannot_reach_prediction_executor(core, snapshot, bridge):
    snapshot['history'][0]['value'] = None
    snapshot['missing_mask'][0] = True
    case = core.create_case('alice', snapshot)
    result = desktop_client.call(1234, 'prediction', payload(case))
    assert result['state'] == 'BLOCKED' and result['reason'] == 'MISSING_INPUT'
    assert counts(core)['jobs'] == counts(core)['artifacts'] == 0


def test_prediction_endpoint_auth_body_and_publication_boundaries(core, run, bridge):
    path = f"/runs/{run['id']}/prediction"
    assert bridge.post(path).status_code == 401
    assert bridge.post(path, headers={'Authorization': 'Bearer b'}).status_code == 404
    assert bridge.post(path, headers={'Authorization': 'Bearer a', 'Origin': 'null'}).status_code == 403
    assert bridge.post(path, headers={'Authorization': 'Bearer a'}, json={'values': [1] * 6}).json()['error'] == 'CLIENT_BODY_FORBIDDEN'
    assert counts(core)['jobs'] == 0
    response = bridge.post(path, headers={'Authorization': 'Bearer a'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert bridge.post(f"/runs/{run['id']}/release", headers={'Authorization': 'Bearer a'}).status_code == 409
    core.cancel('alice', run['id'])
    assert bridge.post(path, headers={'Authorization': 'Bearer a'}).json()['error'] == 'CANCELLED'
    assert counts(core)['artifacts'] == 1 and counts(core)['releases'] == 0


def test_prediction_endpoint_research_never_falls_back(core, snapshot, bridge):
    case = core.create_case('alice', snapshot)
    response = bridge.post('/runs', headers={'Authorization': 'Bearer a'}, json={
        'case_id': case['case_id'], 'idempotency_key': 'research'})
    run = response.json()
    assert run['mode'] == 'research'
    response = bridge.post(f"/runs/{run['id']}/prediction", headers={'Authorization': 'Bearer a'})
    assert response.json()['error'] == 'MODEL_NOT_CONFIGURED'
    assert counts(core)['artifacts'] == 0


@pytest.mark.parametrize('bad', [[], {}, {'case_id': '../runs'},
    {'case_id': 'a' * 32, 'snapshot_id': 'b' * 32, 'idempotency_key': 'c' * 32, 'mode': 'research'},
    {'case_id': 'a' * 32, 'snapshot_id': 'b' * 32, 'idempotency_key': '../execute'}])
def test_desktop_prediction_rejects_arbitrary_input_before_transport(monkeypatch, bad):
    def unexpected(*args):
        pytest.fail('Invalid desktop operation reached HTTP')
    monkeypatch.setattr(desktop_client, '_request', unexpected)
    with pytest.raises(GateError, match='DESKTOP_OPERATION_FORBIDDEN'):
        desktop_client.call(1234, 'prediction', bad)
