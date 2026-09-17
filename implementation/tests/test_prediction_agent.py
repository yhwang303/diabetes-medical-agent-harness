"""P03: actual SDK/CLI to controlled prediction tools; all providers are offline mocks."""

from dataclasses import replace
import json
import sqlite3
import time
from threading import Event, Thread
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import pytest

from conftest import allow_data
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.prediction_agent import PredictionAgent, PredictionTools
from medical_harness.sdk_worker import MODEL, PREDICTION_OPERATIONS, TOOL_PREFIX, permitted
from medical_harness.store import Store
from test_report_agent import FaultRunner, completed
from test_sdk import SSE, message_events


def setup(core, run, tmp_path, *, action=None, provider=None, authorize=True):
    if authorize: allow_data(core, run, 'prediction')
    key = tmp_path / 'offline.key'; key.write_text('offline-prediction'); key.chmod(0o600)
    return PredictionAgent(core, key, tmp_path / 'prediction-agent',
        runner=FaultRunner(action) if action else None,
        transport=httpx.MockTransport(provider or (lambda req: httpx.Response(200,
            stream=SSE(message_events({'type': 'text', 'text': 'done'}))))))


def counts(core):
    with core.store.tx() as db:
        return {name: db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                for name in ('jobs', 'artifacts', 'drafts', 'reviews', 'releases')}


def latest_result(body):
    for message in reversed(body['messages']):
        if not isinstance(message.get('content'), list): continue
        for item in reversed(message['content']):
            if item.get('type') != 'tool_result': continue
            content = item['content']
            text = content if isinstance(content, str) else '\n'.join(x.get('text', '') for x in content)
            return json.loads(text)
    raise AssertionError('SDK must return a tool result to the provider')


def test_actual_sdk_requests_reuses_and_queries_real_worker_without_numerical_leak(tmp_path, snapshot):
    core = Core(Store(tmp_path / 'core.sqlite3'), enable_fixtures=True)
    case = core.create_case('alice', dict(snapshot, source_ref='PRIVATE_RAW_SOURCE'))
    other = core.create_case('alice', dict(snapshot, source_ref='OTHER_CASE_SOURCE'))
    run = core.start_run('alice', case['case_id'], 'eval', 'sdk-prediction')
    calls, references = [], []
    def provider(req):
        body = json.loads(req.content); calls.append(body)
        assert {t['name'] for t in body['tools']} == {TOOL_PREFIX + name for name in PREDICTION_OPERATIONS}
        phase = len(calls)
        if phase == 1:
            assert counts(core)['artifacts'] == 0  # The SDK initiates prediction; host did not run it first.
            name, arguments = 'request_prediction', {}
        elif phase in (2, 3):
            previous = latest_result(body); references.append(previous)
            assert previous['prediction_available'] and previous['evidence']['origin'] == 'fixture'
            name, arguments = ('request_prediction', {}) if phase == 2 else ('inspect_prediction', {'job_id': previous['job_id']})
        else:
            references.append(latest_result(body))
            return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'Numerical claims here are not evidence.'})))
        return httpx.Response(200, stream=SSE(message_events({'type': 'tool_use', 'id': 'prediction-' + str(phase),
            'name': TOOL_PREFIX + name, 'input': arguments})))
    agent = setup(core, run, tmp_path, provider=provider)
    try:
        result = agent.run('alice', run['id'])
        assert len(calls) == 4 and references[0] == references[1] == references[2]
        assert counts(core) == {'jobs': 2, 'artifacts': 1, 'drafts': 0, 'reviews': 0, 'releases': 0}
        assert result == agent.run('alice', run['id']) and len(calls) == 4  # HTTP replay cannot rebill.
        for secret in ('PRIVATE_RAW_SOURCE', 'OTHER_CASE_SOURCE', other['case_id'], 'offline-prediction',
                       'source_ref', 'missing_mask', 'action_value'):
            assert secret not in canonical(calls)
        assert all(set(ref) == {'ok', 'job_id', 'status', 'prediction_available', 'clinical_use', 'evidence'} for ref in references)
        assert '"values"' not in canonical(references)
        with core.store.tx() as db:
            artifact = json.loads(db.execute('SELECT body FROM artifacts').fetchone()[0])
            assert artifact['run_id'] == run['id'] and artifact['snapshot_id'] == case['snapshot_id']
            assert result['evidence']['artifact_id'] == artifact['id'] and artifact['origin'] == 'fixture'
            assert db.execute('SELECT count(*) FROM job_dependencies').fetchone()[0] == 1
            assert db.execute('SELECT count(*) FROM job_data_permissions').fetchone()[0] == 2
            starts = [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE kind='worker_started'")]
            stops = [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE kind='worker_stopped'")]
            assert len(starts) == len(stops) == 2
            assert len({x['pid'] for x in starts}) == 2
            assert db.execute("SELECT count(*) FROM sdk_sessions WHERE state!='REMOVED'").fetchone()[0] == 0
        with pytest.raises(GateError): core.release('alice', run['id'])
        (tmp_path / 'prediction-sdk-proof.json').write_text(canonical({'actual_sdk_worker': True,
            'independent_numeric_worker': True, 'distinct_worker_pids': [x['pid'] for x in starts],
            'upstream': 'offline_mock', 'paid_calls': 0, 'mock_requests': 4, 'prediction_executions': 1,
            'request_query_equal': True, 'numeric_origin': artifact['origin'], 'releases': 0,
            'sdk_directory_removed': True, 'raw_input_and_values_absent_from_outbound': True}))
    finally: agent.close(); core.close()


@pytest.mark.parametrize('purpose', [None, 'report', 'medical_review', 'ethics_review'])
def test_prediction_permission_required_before_key_or_job(core, run, tmp_path, purpose):
    if purpose: allow_data(core, run, purpose)
    agent = PredictionAgent(core, tmp_path / 'missing-key', tmp_path / 'must-not-exist')
    try:
        with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'): agent.run('alice', run['id'])
        assert not agent.data_dir.exists() and counts(core)['jobs'] == 0
    finally: agent.close()


def test_owner_cannot_run_or_query_other_case(core, run, snapshot, tmp_path):
    other = core.create_case('alice', snapshot)
    other_run = core.start_run('alice', other['case_id'], 'eval', 'other')
    core.execute_prediction('alice', other_run['id'])
    with core.store.tx() as db:
        other_job = db.execute("SELECT id FROM jobs WHERE run_id=?", (other_run['id'],)).fetchone()[0]
    def action(client, kwargs):
        assert client.post('/tools/inspect_prediction', json={'job_id': other_job}).json()['error'] == 'NOT_FOUND'
        assert client.post('/tools/request_prediction', json={'case_id': other['case_id']}).json()['error'] == 'INVALID_TOOL_ARGUMENTS'
        result = client.post('/tools/request_prediction', json={}).json()
        assert client.post('/tools/inspect_prediction', json={'job_id': result['job_id']}).json() == result
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError, match='NOT_FOUND'): agent.run('bob', run['id'])
        assert agent.run('alice', run['id'])['prediction_available']
        assert core.status('alice', other_run['id'])['currently_valid']
    finally: agent.close()


@pytest.mark.parametrize('payload', [{'values': [123]}, {'run_id': 'a'*32}, {'snapshot_id': 'a'*32},
                                    {'origin': 'model'}, {'producer': 'trusted'}, {'input_digest': 'a'*64}])
def test_request_cannot_supply_input_identity_or_result(core, run, tmp_path, payload):
    def action(client, kwargs):
        result = client.post('/tools/request_prediction', json=payload).json()
        assert result['error'] == 'INVALID_TOOL_ARGUMENTS' and not result['prediction_available']
        assert counts(core)['artifacts'] == 0
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try: assert agent.run('alice', run['id'])['prediction_available']
    finally: agent.close()


@pytest.mark.parametrize('kind', ['cancel', 'snapshot', 'withdraw', 'withdraw_regrant', 'delete', 'parent', 'config'])
def test_inflight_prediction_rechecks_parent_permission_and_snapshot(core, run, snapshot, tmp_path, kind, monkeypatch):
    original = core._fixtures['prediction']
    def changed(request):
        if kind == 'cancel': core.cancel('alice', run['id'])
        elif kind == 'snapshot': core.update_case('alice', run['case_id'], snapshot)
        elif kind in ('withdraw', 'withdraw_regrant'):
            core.data_permissions.update('alice', run['case_id'], {'snapshot_id': run['snapshot_id'],
                'purpose': 'prediction', 'allowed': False, 'policy_version': 'engineering-data-policy-v1'})
            if kind == 'withdraw_regrant': allow_data(core, run, 'prediction')
        elif kind == 'delete':
            core.data_lifecycle.delete('alice', run['case_id'], {'snapshot_id': run['snapshot_id'], 'contract_version': 'engineering-lifecycle-v1'})
        elif kind == 'parent':
            with core.store.tx() as db: db.execute("UPDATE jobs SET status='FENCED' WHERE step='prediction_agent'")
        else:
            from medical_harness import prediction_agent
            monkeypatch.setattr(prediction_agent, 'prediction_configuration', lambda: {'configuration': 'changed'})
        return original.call(request)
    core._fixtures['prediction'] = replace(original, call=changed)
    def action(client, kwargs):
        assert not client.post('/tools/request_prediction', json={}).json()['ok']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert counts(core)['artifacts'] == counts(core)['releases'] == 0
        assert not core.status('alice', run['id'])['currently_valid']
    finally: agent.close()


def test_timeout_has_no_automatic_retry_or_late_acceptance(core, run, tmp_path):
    core.executor_timeout = .05
    original = core._fixtures['prediction']; calls = []
    def slow(request):
        calls.append(1); time.sleep(.15); return original.call(request)
    core._fixtures['prediction'] = replace(original, call=slow)
    def action(client, kwargs):
        first = client.post('/tools/request_prediction', json={}).json()
        assert first['error'] == 'EXECUTOR_TIMEOUT'
        second = client.post('/tools/request_prediction', json={}).json()
        assert second['status'] == 'FAILED' and not second['prediction_available']
        assert client.post('/tools/inspect_prediction', json={'job_id': second['job_id']}).json() == second
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError, match='EXECUTOR_TIMEOUT'): agent.run('alice', run['id'])
        time.sleep(.16)
        assert len(calls) == 1 and counts(core)['jobs'] == 2 and counts(core)['artifacts'] == 0
    finally: agent.close()


@pytest.mark.parametrize('event', ['prediction_job_bound', 'artifact_accepted', 'prediction_agent_tool', 'prediction_agent_completed'])
def test_audit_failure_cannot_produce_usable_prediction(core, run, tmp_path, monkeypatch, event):
    original = core._event
    def fail(db, run_id, kind, **details):
        if kind == event: raise sqlite3.OperationalError('private audit detail')
        return original(db, run_id, kind, **details)
    monkeypatch.setattr(core, '_event', fail)
    def action(client, kwargs):
        client.post('/tools/request_prediction', json={})
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert not core.status('alice', run['id'])['currently_valid']
        assert counts(core)['releases'] == 0
        if event in ('prediction_job_bound', 'artifact_accepted'): assert counts(core)['artifacts'] == 0
    finally: agent.close()


@pytest.mark.parametrize('fault', ['hash', 'run', 'job', 'origin'])
def test_query_revalidates_authoritative_artifact(core, run, tmp_path, fault):
    def action(client, kwargs):
        result = client.post('/tools/request_prediction', json={}).json()
        with core.store.tx() as db:
            row = db.execute('SELECT * FROM artifacts').fetchone()
            if fault == 'hash': db.execute("UPDATE artifacts SET digest='forged'")
            else:
                from medical_harness.contracts import digest
                body = json.loads(row['body'])
                body[{'run': 'run_id', 'job': 'job_id', 'origin': 'origin'}[fault]] = 'model' if fault == 'origin' else uuid4().hex
                db.execute('UPDATE artifacts SET body=?,digest=?', (canonical(body), digest(body)))
        assert not client.post('/tools/inspect_prediction', json={'job_id': result['job_id']}).json()['ok']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert counts(core)['releases'] == 0
    finally: agent.close()


def test_prediction_tools_cannot_bypass_parent_via_normal_api(core, run, tmp_path):
    def action(client, kwargs):
        with pytest.raises(GateError, match='PREDICTION_SESSION_REQUIRED'): core.execute_prediction('alice', run['id'])
        assert counts(core)['artifacts'] == 0
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try: assert agent.run('alice', run['id'])['prediction_available']
    finally: agent.close()


def test_http_opt_in_auth_and_body_contract(core, run, tmp_path):
    def action(client, kwargs):
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    path = '/runs/' + run['id'] + '/prediction-agent'
    try:
        with TestClient(create_app(core, {'a': 'alice'})) as client:
            assert client.post(path, headers={'Authorization': 'Bearer a'}).json()['error'] == 'AGENT_NOT_CONFIGURED'
        with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'}, prediction_agent=agent)) as client:
            assert client.post(path).status_code == 401
            assert client.post(path, headers={'Authorization': 'Bearer b'}).status_code == 404
            assert client.post(path, headers={'Authorization': 'Bearer a'}, json={}).json()['error'] == 'CLIENT_BODY_FORBIDDEN'
            result = client.post(path, headers={'Authorization': 'Bearer a'})
            assert result.status_code == 200 and result.json()['prediction_available']
            assert result.headers['cache-control'] == 'no-store'
    finally: agent.close()


def test_prediction_profile_has_no_other_roles_or_result_registration():
    for name in ('Bash', 'Read', 'Agent', TOOL_PREFIX + 'prediction_probe', TOOL_PREFIX + 'release_probe',
                 TOOL_PREFIX + 'submit_review', TOOL_PREFIX + 'submit_report_proposal', TOOL_PREFIX + 'execute_rl',
                 TOOL_PREFIX + 'register_prediction', TOOL_PREFIX + 'data_permission_set'):
        assert not permitted(name, {}, profile='prediction')
    assert permitted(TOOL_PREFIX + 'request_prediction', {}, profile='prediction')
    assert not permitted(TOOL_PREFIX + 'request_prediction', {}, profile='prediction', child=True)
    for profile in ('report', 'review', 'smoke'):
        assert not permitted(TOOL_PREFIX + 'request_prediction', {}, profile=profile)


def test_missing_key_is_structured_and_does_not_create_job(core, run, tmp_path):
    allow_data(core, run, 'prediction')
    agent = PredictionAgent(core, tmp_path / 'missing.key', tmp_path / 'agent')
    try:
        with pytest.raises(GateError, match='MODEL_NOT_CONFIGURED'): agent.run('alice', run['id'])
        assert counts(core)['jobs'] == 0
    finally: agent.close()


def test_research_mode_never_falls_back_to_fixture(core, snapshot, tmp_path):
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'research', 'research')
    def action(client, kwargs):
        assert client.post('/tools/request_prediction', json={}).json()['error'] == 'MODEL_NOT_CONFIGURED'
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert counts(core)['artifacts'] == 0
    finally: agent.close()


@pytest.mark.parametrize('action', ['cancel', 'withdraw', 'timeout'])
def test_actual_sdk_and_hanging_numeric_worker_are_reaped(tmp_path, snapshot, action):
    from concurrent.futures import ThreadPoolExecutor
    from test_workers import ProbeRunner, alive
    numerical = ProbeRunner('hang', tmp_path / 'numerical.json')
    core = Core(Store(tmp_path / 'process.sqlite3'), enable_fixtures=True,
                executor_runner=numerical, executor_timeout=.5 if action == 'timeout' else 5)
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'eval', 'process')
    requests = []
    def provider(req):
        requests.append(req)
        content = ({'type': 'tool_use', 'id': 'start-prediction', 'name': TOOL_PREFIX + 'request_prediction', 'input': {}}
                   if len(requests) == 1 else {'type': 'text', 'text': 'stopped'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    agent = setup(core, run, tmp_path, provider=provider)
    try:
        with ThreadPoolExecutor() as pool:
            future = pool.submit(agent.run, 'alice', run['id'])
            deadline = time.monotonic() + 10
            while not numerical.path.exists() and time.monotonic() < deadline: time.sleep(.01)
            assert numerical.path.exists()
            pid = json.loads(numerical.path.read_text())['pid']
            if action == 'cancel': core.cancel('alice', run['id'])
            if action == 'withdraw':
                core.data_permissions.update('alice', case['case_id'], {'snapshot_id': case['snapshot_id'],
                    'purpose': 'prediction', 'allowed': False, 'policy_version': 'engineering-data-policy-v1'})
            with pytest.raises(GateError): future.result(timeout=15)
        assert not alive(pid) and not numerical._active and not agent.runner._active
        assert counts(core)['artifacts'] == counts(core)['releases'] == 0
        with core.store.tx() as db:
            assert db.execute("SELECT count(*) FROM sdk_sessions WHERE state!='REMOVED'").fetchone()[0] == 0
        (tmp_path / 'prediction-fault-proof.json').write_text(canonical({'actual_sdk_worker': True,
            'numeric_worker': 'isolated_hang_probe', 'fault': action, 'numeric_pid': pid,
            'workers_reaped': True, 'artifacts': 0, 'paid_calls': 0, 'sdk_directory_removed': True}))
    finally: agent.close(); core.close()


@pytest.mark.parametrize('record', ['parent_permission', 'child_permission', 'dependency'])
def test_missing_inflight_binding_is_not_treated_as_plain_fixture(core, run, tmp_path, record):
    original = core._fixtures['prediction']
    def corrupt(request):
        with core.store.tx() as db:
            if record == 'dependency': db.execute('DELETE FROM job_dependencies')
            else:
                step = 'prediction_agent' if record == 'parent_permission' else 'prediction'
                db.execute('DELETE FROM job_data_permissions WHERE job_id IN (SELECT id FROM jobs WHERE step=?)', (step,))
        return original.call(request)
    core._fixtures['prediction'] = replace(original, call=corrupt)
    def action(client, kwargs):
        assert not client.post('/tools/request_prediction', json={}).json()['ok']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert counts(core)['artifacts'] == 0
    finally: agent.close()


def test_sdk_failure_after_numeric_acceptance_invalidates_run(core, run, tmp_path):
    def action(client, kwargs):
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        return {'results': [{'is_error': True, 'terminal_reason': 'completed'}]}
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError, match='PREDICTION_AGENT_INCOMPLETE'): agent.run('alice', run['id'])
        assert counts(core)['artifacts'] == 1 and not core.status('alice', run['id'])['currently_valid']
        with pytest.raises(GateError): core.execute_models('alice', run['id'])
        with pytest.raises(GateError): core.release('alice', run['id'])
    finally: agent.close()


def test_local_http_to_sdk_mcp_and_numeric_worker(tmp_path, snapshot):
    import socket
    import uvicorn
    calls = []
    def provider(req):
        body = json.loads(req.content); calls.append(body)
        if len(calls) == 1: name, raw = 'request_prediction', {}
        elif len(calls) == 2: name, raw = 'inspect_prediction', {'job_id': latest_result(body)['job_id']}
        else: return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'done'})))
        return httpx.Response(200, stream=SSE(message_events({'type': 'tool_use', 'id': 'http-pred-' + str(len(calls)),
            'name': TOOL_PREFIX + name, 'input': raw})))
    core = Core(Store(tmp_path / 'http.sqlite3'), enable_fixtures=True)
    key = tmp_path / 'offline.key'; key.write_text('offline-only'); key.chmod(0o600)
    agent = PredictionAgent(core, key, tmp_path / 'agent', transport=httpx.MockTransport(provider))
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(core, {'offline-token': 'alice'}, prediction_agent=agent),
                                          access_log=False, log_level='error'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True); thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline: time.sleep(.02)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=20,
                          headers={'Authorization': 'Bearer offline-token'}) as client:
            case = client.post('/cases', json=snapshot).json()
            run = client.post('/runs', json={'case_id': case['case_id'], 'mode': 'eval',
                'expected_snapshot_id': case['snapshot_id'], 'idempotency_key': uuid4().hex}).json()
            path = '/runs/' + run['id'] + '/prediction-agent'
            assert client.post(path).json()['error'] == 'DATA_PERMISSION_REQUIRED' and not calls
            grant = client.put('/cases/' + case['case_id'] + '/data-permissions', json={
                'snapshot_id': case['snapshot_id'], 'purpose': 'prediction', 'allowed': True,
                'policy_version': 'engineering-data-policy-v1'})
            assert grant.json()['permissions']['prediction']
            result = client.post(path)
            assert result.status_code == 200 and result.json()['prediction_available']
            assert client.post(path).json() == result.json() and len(calls) == 3
            assert counts(core) == dict(jobs=2, artifacts=1, drafts=0, reviews=0, releases=0)
            (tmp_path / 'prediction-http-proof.json').write_text(canonical({'actual_local_http': True,
                'actual_sdk_mcp': True, 'independent_numeric_worker': True, 'paid_calls': 0,
                'unauthorized_dispatches': 0, 'mock_requests': 3, 'http_retry_reuses_result': True,
                'artifacts': 1, 'origin': 'fixture', 'releases': 0}))
    finally:
        server.should_exit = True; thread.join(timeout=3); sock.close(); agent.close(); core.close()
    assert not thread.is_alive()
