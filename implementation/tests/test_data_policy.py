"""Offline data-flow and consent checks; no provider account or real patient data."""

import json
import sqlite3
import socket
import time
from threading import Thread
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from conftest import allow_data
from medical_harness.api import create_app
from medical_harness.agent_tasks import AgentTasks
from medical_harness import desktop_client
from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core
from medical_harness.data_policy import POLICY_VERSION, data_policy_catalog
from medical_harness.report_agent import ReportAgent
from medical_harness.sdk_worker import MODEL, TOOL_PREFIX, permitted
from medical_harness.store import Store
from test_report_agent import setup_agent, FaultRunner, completed, VALID
from test_review_agent import prepare, FaultRunner as ReviewRunner, count
from test_sdk import request, SSE, message_events


def change(core, case, *, purpose='report', allowed=False):
    return core.data_permissions.update('alice', case['case_id'], {
        'snapshot_id': case['snapshot_id'], 'purpose': purpose, 'allowed': allowed, 'policy_version': POLICY_VERSION})


def test_default_denial_precedes_key_files_and_job_creation(core, run, tmp_path):
    core.execute_models('alice', run['id'])
    agent = ReportAgent(core, tmp_path / 'key-must-not-be-read', tmp_path / 'no-directory')
    with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'):
        agent.generate('alice', run['id'])
    assert not agent.data_dir.exists()
    assert count(core, 'job_data_permissions') == count(core, 'drafts') == 0
    assert core.data_permissions.read('alice', run['case_id'])['permissions'] == {
        'report': False, 'medical_review': False, 'ethics_review': False, 'prediction': False, 'rl': False}
    agent.runner.close()


@pytest.mark.parametrize('fault,code', [('owner', 'NOT_FOUND'), ('snapshot', 'SNAPSHOT_CHANGED'),
    ('case', 'DATA_PERMISSION_REQUIRED'), ('purpose', 'DATA_PERMISSION_REQUIRED'),
    ('withdraw', 'DATA_PERMISSION_REQUIRED'), ('policy', 'DATA_PERMISSION_REQUIRED')])
def test_permission_is_bound_to_exact_case_snapshot_purpose(core, run, snapshot, fault, code, monkeypatch):
    allow_data(core, run)
    owner, run_id, purpose = 'alice', run['id'], 'report'
    if fault == 'owner': owner = 'bob'
    elif fault == 'snapshot':
        core.update_case('alice', run['case_id'], snapshot)
    elif fault == 'case':
        other = core.create_case('alice', snapshot)
        run_id = core.start_run('alice', other['case_id'], 'eval', 'other')['id']
    elif fault == 'purpose': purpose = 'medical_review'
    elif fault == 'withdraw': change(core, run)
    else:
        import medical_harness.data_policy as module
        monkeypatch.setattr(module, 'data_policy_catalog', lambda: {'version': 'changed'})
    with pytest.raises(GateError, match=code): core.data_permissions.authorize_run(owner, run_id, purpose)


def test_idempotence_reauthorization_and_restart(core, run):
    allow_data(core, run)
    first = core.data_permissions.authorize_run('alice', run['id'], 'report')
    allow_data(core, run)
    assert core.data_permissions.authorize_run('alice', run['id'], 'report') == first
    change(core, run)
    allow_data(core, run)
    assert core.data_permissions.authorize_run('alice', run['id'], 'report')['revision'] == first['revision'] + 2
    fresh = Core(Store(core.store.path), enable_fixtures=True)
    try:
        assert not fresh.data_permissions.read('alice', run['case_id'])['permissions']['report']
        with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'):
            fresh.data_permissions.authorize_run('alice', run['id'], 'report')
    finally: fresh.close()


def test_database_backup_cannot_restore_outbound_permission(core, run, tmp_path):
    allow_data(core, run)
    backup = tmp_path / 'restored.sqlite3'
    with sqlite3.connect(core.store.path) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    restored = Core(Store(backup), enable_fixtures=True)
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'):
            restored.data_permissions.authorize_run('alice', run['id'], 'report')
        assert count(restored, 'data_permissions') == 1  # Preserved record is not renewed consent.
    finally: restored.close()


@pytest.mark.parametrize('source', ['historical', 'simulation'])
def test_current_consent_does_not_unlock_real_data(core, snapshot, source):
    snapshot['source'] = source
    case = core.create_case('alice', snapshot)
    with pytest.raises(GateError, match='DATA_SCOPE_NOT_SUPPORTED'): allow_data(core, case)
    assert not core.data_permissions.read('alice', case['case_id'])['permissions']['report']


@pytest.mark.parametrize('mutation', [{'allowed': 'true'}, {'purpose': 'observation'},
    {'policy_version': 'old'}, {'raw_data': 'DO_NOT_ECHO'}, {'case_id': 'another'}])
def test_api_rejects_unapproved_fields_without_echo(core, run, mutation):
    body = {'snapshot_id': run['snapshot_id'], 'purpose': 'report', 'allowed': True, 'policy_version': POLICY_VERSION} | mutation
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        url = f"/cases/{run['case_id']}/data-permissions"
        assert client.put(url, json=body).status_code == 401
        assert client.put(url, headers={'Authorization': 'Bearer b'}, json={k:v for k,v in body.items() if k != 'case_id'}).status_code in (404, 422)
        response = client.put(url, headers={'Authorization': 'Bearer a'}, json=body)
        assert response.status_code == 422 and response.json() == {'error': 'INVALID_SCHEMA'}
    assert count(core, 'data_permissions') == 0
    with core.store.tx() as db:
        assert 'DO_NOT_ECHO' not in canonical([row['body'] for row in db.execute('SELECT body FROM events')])


def test_audit_failure_rolls_back_permission(core, run, monkeypatch):
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == 'data_permission_changed': raise sqlite3.OperationalError('private detail')
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    with pytest.raises(sqlite3.OperationalError): allow_data(core, run)
    assert count(core, 'data_permissions') == 0


def test_http_permission_owner_isolation_and_desktop_mapping(core, run, monkeypatch):
    body = {'snapshot_id': run['snapshot_id'], 'purpose': 'report', 'allowed': True, 'policy_version': POLICY_VERSION}
    path = f"/cases/{run['case_id']}/data-permissions"
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        for method in ('get', 'put'):
            response = client.request(method, path, headers={'Authorization': 'Bearer b'},
                                      **({'json': body} if method == 'put' else {}))
            assert response.status_code == 404
        assert client.put(path, headers={'Authorization': 'Bearer a'}, json=body).json()['permissions']['report']
        assert client.get(path, headers={'Authorization': 'Bearer a'}).json()['permissions']['report']
    calls = []
    monkeypatch.setattr(desktop_client, '_request', lambda *args: calls.append(args) or {'ok': True})
    desktop_client.call(8787, 'data-permissions', {'case_id': run['case_id']})
    desktop_client.call(8787, 'data-permission-set', {'case_id': run['case_id'], **body})
    assert calls[0] == (8787, 'GET', path, None)
    assert calls[1][:3] == (8787, 'PUT', path) and json.loads(calls[1][3]) == body
    with pytest.raises(GateError):
        desktop_client.call(8787, 'data-permission-set', {'case_id': '../another', **body})
    assert len(calls) == 2


@pytest.mark.parametrize('renew', [False, True])
def test_withdrawal_blocks_next_upstream_and_staged_report(core, run, tmp_path, renew):
    calls = []
    def provider(req):
        calls.append(req)
        return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'done'})))
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        change(core, run)
        if renew: allow_data(core, run)
        response = client.post('/v1/messages', json=request())
        assert response.status_code == 400
        assert response.json()['error']['message'] == ('DATA_PERMISSION_CHANGED' if renew else 'DATA_PERMISSION_REQUIRED')
        assert client.post('/tools/inspect_report_contract', json={}).json()['ok'] is False
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action), provider=provider)
    try:
        with pytest.raises(GateError): agent.generate('alice', run['id'])
        assert len(calls) == 1
        assert json.loads((agent.data_dir / 'budget.json').read_text())['requests'] == 1
        assert count(core, 'drafts') == count(core, 'releases') == 0
    finally: agent.runner.close()


def test_review_permission_is_separate_and_precedes_key(tmp_path, snapshot):
    core, reviewer, run, _ = prepare(tmp_path, snapshot)
    change(core, run, purpose='medical_review')
    reviewer.key_path.unlink()
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'): core.review('alice', run['id'], 'medical')
        assert reviewer.evidence == [] and not reviewer.data_dir.exists()
        assert count(core, 'reviews') == 0
    finally: core.close()


def test_review_withdrawal_after_staging_rejects_late_result(tmp_path, snapshot):
    def action(client, output, kwargs):
        assert client.post('/tools/submit_review', json=output).json()['ok']
        change(core, run, purpose='medical_review')
        return completed()
    core, reviewer, run, _ = prepare(tmp_path, snapshot, runner=ReviewRunner(action))
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'): core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == count(core, 'releases') == 0
    finally: core.close()


def test_old_queued_permission_cannot_use_a_new_grant(core, run, tmp_path):
    allow_data(core, run)
    saved = core.data_permissions.authorize_run('alice', run['id'], 'report')
    change(core, run)
    allow_data(core, run)
    agent = ReportAgent(core, tmp_path / 'missing', tmp_path / 'no-directory')
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_CHANGED'):
            agent.generate('alice', run['id'], expected_permission=saved)
        assert not agent.data_dir.exists()
    finally: agent.runner.close()


@pytest.mark.parametrize('fault', ['hash', 'purpose'])
def test_bound_permission_corruption_prevents_registration(core, run, tmp_path, fault):
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        with core.store.tx() as db:
            row = db.execute('SELECT * FROM job_data_permissions').fetchone()
            if fault == 'hash': db.execute("UPDATE job_data_permissions SET digest='wrong'")
            else:
                body = json.loads(row['body'])
                body['purpose'] = 'ethics_review'
                db.execute('UPDATE job_data_permissions SET body=?,digest=?', (canonical(body), digest(body)))
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_'):
            agent.generate('alice', run['id'])
        assert count(core, 'drafts') == 0
    finally: agent.runner.close()


def test_stream_rechecks_permission_after_dispatch(core, run, tmp_path):
    class WithdrawStream(httpx.SyncByteStream):
        def __iter__(self):
            change(core, run)
            yield b'event: ping\ndata: {"type":"ping"}\n\n'
    def provider(req): return httpx.Response(200, stream=WithdrawStream())
    def action(client, kwargs): return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action), provider=provider)
    try:
        with pytest.raises(GateError): agent.generate('alice', run['id'])
        assert {'error': 'DATA_PERMISSION_REQUIRED'} in agent.evidence['provider']
        assert count(core, 'drafts') == 0
    finally: agent.runner.close()


def test_actual_sdk_report_sends_no_raw_snapshot_or_other_case(core, snapshot, tmp_path):
    snapshot['source_ref'] = 'PRIVATE_SOURCE_MARKER_NOT_FOR_LLM'
    case = core.create_case('alice', snapshot)
    other = core.create_case('alice', dict(snapshot, source_ref='OTHER_CASE_SECRET'))
    run = core.start_run('alice', case['case_id'], 'eval', 'outbound-test')
    requests = []
    def provider(req):
        body = json.loads(req.content)
        requests.append(body)
        operations = [('inspect_report_contract', {}), ('submit_report_proposal', VALID)]
        content = ({'type': 'tool_use', 'id': 'call-' + str(len(requests)),
                    'name': TOOL_PREFIX + operations[len(requests)-1][0], 'input': operations[len(requests)-1][1]}
                   if len(requests) <= 2 else {'type': 'text', 'text': 'done'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    agent = setup_agent(core, run, tmp_path, provider=provider)
    try:
        assert agent.generate('alice', run['id'])['state'] == 'DRAFT_READY'
        assert len(requests) == 3 and all(body['model'] == MODEL for body in requests)
        for secret in [snapshot['source_ref'], 'OTHER_CASE_SECRET', other['case_id'], 'offline-secret', 'source_ref', 'missing_mask']:
            assert secret not in canonical(requests)
        for profile in ('report', 'review'):
            assert not permitted(TOOL_PREFIX + 'data_permission_set', {}, profile=profile)
        assert data_policy_catalog()['observation'] == {'enabled': False, 'raw_upload': False}
    finally: agent.runner.close()


def test_local_http_desktop_task_authorize_run_withdraw(core, snapshot, tmp_path):
    calls = []
    def provider(req):
        body = json.loads(req.content)
        calls.append(body)
        steps = [('inspect_report_contract', {}), ('submit_report_proposal', VALID)]
        content = ({'type': 'tool_use', 'id': 'http-call-' + str(len(calls)),
                    'name': TOOL_PREFIX + steps[len(calls)-1][0], 'input': steps[len(calls)-1][1]}
                   if len(calls) <= 2 else {'type': 'text', 'text': 'done'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    key = tmp_path / 'offline.key'
    key.write_text('offline-only'); key.chmod(0o600)
    reporter = ReportAgent(core, key, tmp_path / 'report', transport=httpx.MockTransport(provider))
    tasks = AgentTasks(core, reporter)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(core, {'offline-token': 'alice'}, agent_tasks=tasks),
                                         access_log=False, log_level='error'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline: time.sleep(.02)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False,
                          headers={'Authorization': 'Bearer offline-token'}) as client:
            case = client.post('/cases', json=snapshot).json()
            payload = dict(case, idempotency_key=uuid4().hex)
            assert client.post('/agent/tasks', json=payload).json()['error'] == 'DATA_PERMISSION_REQUIRED'
            assert not calls
            path = f"/cases/{case['case_id']}/data-permissions"
            grant = {'snapshot_id': case['snapshot_id'], 'purpose': 'report', 'allowed': True, 'policy_version': POLICY_VERSION}
            assert client.put(path, json=grant).json()['permissions']['report']
            task = client.post('/agent/tasks', json=payload).json()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = client.get('/agent/tasks/' + task['id']).json()
                if not state['active']: break
                time.sleep(.05)
            assert state['state'] == 'SUCCEEDED'
            assert len(calls) == 3 and count(core, 'drafts') == 1
            assert not client.put(path, json=dict(grant, allowed=False)).json()['permissions']['report']
            assert client.post('/agent/tasks', json=dict(payload, idempotency_key=uuid4().hex)).json()['error'] == 'DATA_PERMISSION_REQUIRED'
            assert len(calls) == 3 and count(core, 'releases') == 0
            (tmp_path / 'http-proof.json').write_text(canonical({'real_local_http': True,
                'actual_sdk_worker': True, 'upstream': 'offline_mock_only', 'paid_calls': 0,
                'mock_requests': len(calls), 'before_grant_denied': True, 'after_withdraw_denied': True,
                'task_state': state['state'], 'drafts': 1, 'reviews': 0, 'releases': 0}))
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        tasks.close()
        sock.close()
    assert not thread.is_alive()
