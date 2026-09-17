import json
import sqlite3
import time
from threading import Event
from uuid import uuid4

import httpx
import pytest
from conftest import allow_data
from fastapi.testclient import TestClient

from medical_harness.agent_tasks import AgentTasks
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical
from medical_harness.report_agent import ReportAgent
from medical_harness import desktop_client
from medical_harness.sdk_worker import TOOL_PREFIX
from medical_harness.workers import group_rss
from test_report_agent import FaultRunner, VALID, completed
from test_sdk import SSE, message_events


def wait(tasks, task, *, terminal=True):
    end = time.monotonic() + 15
    while time.monotonic() < end:
        result = tasks.status('alice', task['id'])
        if (not result['active']) if terminal else (result['phase'] == 'report_agent'):
            return result
        time.sleep(.02)
    pytest.fail('task did not reach expected state')


@pytest.fixture
def setup(core, snapshot, tmp_path):
    key = tmp_path / 'offline.key'
    key.write_text('offline-secret'); key.chmod(0o600)
    def accept(client, kwargs):
        assert client.post('/tools/inspect_report_contract', json={}).json()['ok']
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        return completed()
    provider = httpx.MockTransport(lambda req: httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'done'}))))
    reporter = ReportAgent(core, key, tmp_path / 'agent', runner=FaultRunner(accept), transport=provider)
    tasks = AgentTasks(core, reporter)
    case = core.create_case('alice', snapshot)
    allow_data(core, case)
    request = dict(case, idempotency_key=uuid4().hex)
    yield tasks, request
    tasks.close()


def test_async_task_idempotency_status_and_no_payload_leak(setup, core):
    tasks, request = setup
    assert tasks.latest('alice', request['case_id']) is None
    task = tasks.start('alice', request)
    assert tasks.start('alice', request)['id'] == task['id']
    result = wait(tasks, task)
    assert result['state'] == 'SUCCEEDED' and result['run_state'] == 'DRAFT_READY'
    assert result['currently_valid'] and not result['active']
    assert tasks.latest('alice', request['case_id'])['id'] == task['id']
    assert [item['tool'] for item in result['progress'] if item['kind'] == 'report_agent_tool'] == ['inspect_report_contract', 'submit_report_proposal']
    assert not any(s in canonical(result) for s in ['offline-secret', 'capability', 'action_value', 'payload', 'sections'])
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM agent_tasks').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM drafts').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM reviews').fetchone()[0] == db.execute('SELECT count(*) FROM releases').fetchone()[0] == 0
    with pytest.raises(GateError, match='REVIEW_REQUIRED'):
        core.release('alice', task['run_id'])


@pytest.mark.parametrize('fault,expected', [('source','FIXTURE_REQUIRES_SYNTHETIC_CASE'), ('missing','MISSING_INPUT'),
    ('snapshot','SNAPSHOT_CHANGED'), ('owner','NOT_FOUND'), ('key','AGENT_NOT_CONFIGURED'), ('fixtures','FIXTURES_DISABLED')])
def test_preflight_blocks_before_execution(setup, core, snapshot, fault, expected):
    tasks, request = setup
    if fault in ('source', 'missing'):
        if fault == 'source': snapshot['source'] = 'historical'
        else: snapshot['history'][0]['value'] = None; snapshot['missing_mask'][0] = True
        case = core.create_case('alice', snapshot)
        request.update(case)
    elif fault == 'snapshot': core.update_case('alice', request['case_id'], snapshot)
    elif fault == 'key': tasks.reporter.key_path.unlink()
    elif fault == 'fixtures': core.enable_fixtures = False
    with pytest.raises(GateError, match=expected):
        tasks.start('bob' if fault == 'owner' else 'alice', request)
    assert tasks.reporter.evidence == {}
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0


def test_cancel_staged_output_and_repeated_cancel_are_safe(setup, core):
    tasks, request = setup
    staged, stop = Event(), Event()
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        staged.set(); assert stop.wait(5)
        return completed()
    tasks.reporter.runner = FaultRunner(action)
    task = tasks.start('alice', request)
    try:
        assert staged.wait(5)
        with pytest.raises(GateError, match='EXECUTOR_BUSY'):
            tasks.start('alice', dict(request, idempotency_key=uuid4().hex))
        assert tasks.cancel('alice', task['id'])['state'] == 'CANCELLING'
        tasks.cancel('alice', task['id'])
    finally:
        stop.set()
    result = wait(tasks, task)
    assert result['state'] == 'CANCELLED' and not result['currently_valid']
    assert tasks.start('alice', request)['state'] == 'CANCELLED'
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM drafts').fetchone()[0] == 0


def test_restart_keeps_completed_task_and_invalidates_interrupted_task(setup, core):
    tasks, request = setup
    done = wait(tasks, tasks.start('alice', request))
    tasks.close()
    restored = AgentTasks(core, tasks.reporter)
    try:
        assert restored.status('alice', done['id'])['state'] == 'SUCCEEDED'
    finally:
        restored.close()
    with core.store.tx() as db:
        db.execute("UPDATE agent_tasks SET state='RUNNING'")
    restored = AgentTasks(core, tasks.reporter)
    try:
        result = restored.status('alice', done['id'])
        assert result['state'] == 'FAILED' and result['reason'] == 'SERVICE_RESTARTED'
        assert not result['currently_valid']
        assert restored.start('alice', request)['id'] == done['id']
    finally:
        restored.close()


def test_configuration_reads_never_call_provider_and_snapshot_change_revokes_success(setup, core, snapshot):
    tasks, request = setup
    for _ in range(3):
        assert tasks.configuration()['credential_ready']
        assert tasks.latest('alice', request['case_id']) is None
    assert tasks.reporter.evidence == {}
    task = wait(tasks, tasks.start('alice', request))
    core.update_case('alice', request['case_id'], snapshot)
    result = tasks.status('alice', task['id'])
    assert not result['currently_valid'] and result['invalid_reason'] == 'SNAPSHOT_CHANGED'
    with pytest.raises(GateError, match='IDEMPOTENCY_CONFLICT'):
        tasks.start('alice', dict(request, snapshot_id='a'*32))


def test_start_audit_failure_does_not_dispatch(setup, core, monkeypatch):
    tasks, request = setup
    original = core._event
    def audit(db, run_id, kind, **details):
        if kind == 'agent_task_created': raise sqlite3.OperationalError('unavailable')
        return original(db, run_id, kind, **details)
    monkeypatch.setattr(core, '_event', audit)
    with pytest.raises(sqlite3.OperationalError):
        tasks.start('alice', request)
    assert tasks.reporter.evidence == {}
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM agent_tasks').fetchone()[0] == 0


def test_api_and_desktop_bridge_authentication_and_fixed_operations(setup, core, monkeypatch):
    tasks, request = setup
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'}, agent_tasks=tasks)) as client:
        headers = {'Authorization': 'Bearer a'}
        def transport(port, method, path, body):
            response = client.request(method, path, content=body, headers=headers)
            if response.is_error: raise GateError(response.json()['error'])
            return response.json()
        monkeypatch.setattr(desktop_client, '_request', transport)
        assert client.post('/agent/tasks', json=request).status_code == 401
        assert client.post('/agent/tasks', headers=dict(headers, Origin='null'), json=request).status_code == 403
        assert client.post('/agent/tasks', headers=headers, json=dict(request, prompt='publish 999 units')).status_code == 422
        result = desktop_client.call(1234, 'agent-start', request)
        task_id = result['id']
        wait(tasks, result)
        assert client.get('/agent/tasks/'+task_id, headers={'Authorization':'Bearer b'}).status_code == 404
        assert client.post('/agent/tasks/'+task_id+'/cancel', headers=headers, json={'verdict':'pass'}).status_code == 422
        assert desktop_client.call(1234, 'agent-status', {'task_id':task_id})['state'] == 'SUCCEEDED'
        assert desktop_client.call(1234, 'agent-latest', {'case_id':request['case_id']})['id'] == task_id
        assert desktop_client.call(1234, 'agent-cancel', {'task_id':task_id})['state'] == 'SUCCEEDED'
        assert desktop_client.call(1234, 'agent-configuration', {})['model'] == 'deepseek-flash'
        for name in ('agent-status','agent-cancel','agent-latest'):
            with pytest.raises(GateError, match='DESKTOP_OPERATION_FORBIDDEN'):
                desktop_client.call(1234, name, {'task_id':'../release'})


@pytest.mark.parametrize('stop_kind', ['cancel', 'shutdown'])
def test_real_sdk_task_staged_then_cancel_or_shutdown_reaps_worker(setup, core, stop_kind):
    tasks, request = setup
    from medical_harness.workers import WorkerRunner, WorkerLimits
    tasks.reporter.runner = WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
        file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)
    staged, stop = Event(), Event()
    calls = []
    class Stalled(httpx.SyncByteStream):
        def __iter__(self):
            while not stop.wait(.05): pass
            yield b''
    def provider(req):
        calls.append(True)
        if len(calls) == 3:
            staged.set()
            return httpx.Response(200, stream=Stalled())
        content = {'type':'tool_use','id':str(len(calls)), 'name':TOOL_PREFIX + ('inspect_report_contract' if len(calls)==1 else 'submit_report_proposal'),
                   'input': {} if len(calls)==1 else VALID}
        return httpx.Response(200, stream=SSE(message_events(content)))
    tasks.reporter.transport = httpx.MockTransport(provider)
    task = tasks.start('alice', request)
    try:
        assert staged.wait(10)
        if stop_kind == 'cancel': tasks.cancel('alice', task['id'])
        else: tasks.close()
        assert wait(tasks, task)['state'] == 'CANCELLED'
        events = tasks.reporter.evidence['workers']
        assert events[-1]['exit_code'] == -9
        assert all(group_rss(e['pid']) == 0 for e in events if e['kind'] == 'worker_started')
        with core.store.tx() as db:
            assert db.execute('SELECT count(*) FROM drafts').fetchone()[0] == 0
    finally:
        stop.set()


def test_finish_audit_failure_is_unavailable_not_permanently_running(setup, core, monkeypatch):
    tasks, request = setup
    original = core._event
    def audit(db, run_id, kind, **details):
        if kind == 'agent_task_finished': raise sqlite3.OperationalError('unavailable')
        return original(db, run_id, kind, **details)
    monkeypatch.setattr(core, '_event', audit)
    task = tasks.start('alice', request)
    with pytest.raises(sqlite3.OperationalError):
        tasks._submitted[1].result(timeout=10)
    with pytest.raises(GateError, match='STORAGE_UNAVAILABLE'):
        tasks.status('alice', task['id'])
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM releases').fetchone()[0] == 0


@pytest.mark.parametrize('kind,code', [('incomplete','REPORT_AGENT_INCOMPLETE'), ('error','REPORT_AGENT_FAILED')])
def test_async_provider_failure_is_terminal_and_sanitized(setup, core, kind, code):
    tasks, request = setup
    def failure(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        if kind == 'error': raise RuntimeError('PRIVATE_PROVIDER_TEXT')
        return {'results':[{'is_error':True}]}
    tasks.reporter.runner = FaultRunner(failure)
    task = tasks.start('alice', request)
    with pytest.raises(GateError, match='NOT_FOUND'):
        tasks.cancel('bob', task['id'])
    result = wait(tasks, task)
    assert result['state'] == 'FAILED' and result['reason'] == code
    assert 'PRIVATE_PROVIDER_TEXT' not in canonical(result)
    with core.store.tx() as db:
        assert db.execute('SELECT count(*) FROM drafts').fetchone()[0] == 0
    tasks.close()
    with pytest.raises(GateError, match='CORE_SHUTTING_DOWN'):
        tasks.start('alice', dict(request, idempotency_key=uuid4().hex))
