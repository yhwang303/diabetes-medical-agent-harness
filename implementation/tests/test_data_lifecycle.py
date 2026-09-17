"""Deletion and SDK retention on isolated synthetic ledgers; no real provider calls."""

import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from conftest import allow_data
from medical_harness import data_lifecycle as lifecycle, desktop_client
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.store import Store
from test_core import reviewed, count
from test_report_agent import setup_agent, FaultRunner, completed, VALID
from test_review_agent import prepare, FaultRunner as ReviewRunner


def delete(core, case, owner='alice'):
    return core.data_lifecycle.delete(owner, case['case_id'], {
        'snapshot_id': case['snapshot_id'], 'contract_version': 'engineering-lifecycle-v1'})


def sdk_job(core, run):
    core.execute_models('alice', run['id'])
    return core.create_draft_job('alice', run['id'])['job_id']


def test_delete_all_content_versions_and_release_preserves_other_case(core, snapshot):
    first = core.import_case('alice', {'file_name': 'PRIVATE_FILENAME.json',
        'content': canonical(dict(snapshot, source_ref='PRIVATE_SOURCE'))})
    case = core.update_case('alice', first['case_id'], dict(snapshot, source_ref='PRIVATE_REPLACEMENT'))
    run = core.start_run('alice', case['case_id'], 'eval', 'PRIVATE_REQUEST')
    reviewed(core, run['id'])
    release = core.release('alice', run['id'])
    allow_data(core, case)
    other = core.create_case('alice', dict(snapshot, source_ref='OTHER_CASE_SOURCE'))
    other_before = core.input_timeline('alice', other['case_id'])
    result = delete(core, case)
    assert result['state'] == 'LOCAL_CONTENT_REMOVED' and result['local_content_removed']
    assert core.input_timeline('alice', other['case_id']) == other_before
    assert [row['case_id'] for row in core.list_cases('alice')] == [other['case_id']]
    assert delete(core, case) == result
    assert not core.status('alice', run['id'])['currently_valid']
    with pytest.raises(GateError): core.get_release('alice', release['release_id'])
    with core.store.tx() as db:
        for table in ('imports', 'artifacts', 'drafts', 'reviews', 'releases', 'run_profiles',
                      'review_configs', 'data_permissions', 'job_data_permissions'):
            assert db.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM snapshots WHERE case_id=?', (case['case_id'],)).fetchone()[0] == 0
        dump = '\n'.join(db.iterdump())
        assert all(marker not in dump for marker in ('PRIVATE_FILENAME', 'PRIVATE_SOURCE', 'PRIVATE_REPLACEMENT', 'PRIVATE_REQUEST'))
        assert db.execute("SELECT count(*) FROM events WHERE kind='case_content_deleted'").fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == db.execute('SELECT count(*) FROM outbox').fetchone()[0]
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []


@pytest.mark.parametrize('operation', ['timeline', 'evidence', 'update', 'start', 'execute', 'grant', 'draft', 'release'])
def test_tombstone_blocks_every_content_or_execution_path(core, run, snapshot, operation):
    delete(core, run)
    operations = {'timeline': lambda: core.input_timeline('alice', run['case_id']),
        'evidence': lambda: core.case_evidence('alice', run['case_id']),
        'update': lambda: core.update_case('alice', run['case_id'], snapshot),
        'start': lambda: core.start_run('alice', run['case_id'], 'eval', 'new'),
        'execute': lambda: core.execute_models('alice', run['id']),
        'grant': lambda: allow_data(core, run), 'draft': lambda: core.create_draft_job('alice', run['id']),
        'release': lambda: core.release('alice', run['id'])}
    with pytest.raises(GateError, match='CASE_DELETED'): operations[operation]()


def test_delete_strict_owner_snapshot_and_api_no_echo(core, run):
    body = {'snapshot_id': run['snapshot_id'], 'contract_version': 'engineering-lifecycle-v1'}
    url = '/cases/' + run['case_id']
    with TestClient(create_app(core, {'a': 'alice', 'b': 'bob'})) as client:
        assert client.request('DELETE', url, json=body).status_code == 401
        assert client.request('DELETE', url, json=body, headers={'Authorization': 'Bearer b'}).status_code == 404
        assert client.get(url + '/lifecycle', headers={'Authorization': 'Bearer b'}).status_code == 404
        auth = {'Authorization': 'Bearer a'}
        assert client.request('DELETE', url, json=body | {'snapshot_id': uuid4().hex}, headers=auth).json()['error'] == 'SNAPSHOT_CHANGED'
        bad = client.request('DELETE', url, json=body | {'raw': 'SECRET_DO_NOT_ECHO'}, headers=auth)
        assert bad.status_code == 422 and 'SECRET' not in bad.text
        assert client.get(url + '/lifecycle', headers=auth).json()['state'] == 'RETAINED'
        assert client.request('DELETE', url, json=body, headers=auth).json()['local_content_removed']
        assert client.get(url + '/timeline', headers=auth).status_code == 410
        assert client.get(url + '/lifecycle', headers=auth).headers['cache-control'] == 'no-store'


def test_delete_audit_failure_rolls_back_all_content(core, run, monkeypatch):
    core.execute_models('alice', run['id'])
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == 'case_content_deleted': raise sqlite3.OperationalError('private storage detail')
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    with pytest.raises(sqlite3.OperationalError): delete(core, run)
    assert core.data_lifecycle.read('alice', run['case_id'])['state'] == 'RETAINED'
    assert core.status('alice', run['id'])['currently_valid']
    assert count(core, 'artifacts') == 2 and count(core, 'snapshots') == 1


@pytest.mark.parametrize('after_delete', [False, True])
def test_backup_restoration_does_not_restore_permission_and_preserves_receipt(core, run, tmp_path, after_delete):
    allow_data(core, run)
    if after_delete: delete(core, run)
    destination = tmp_path / 'restored.sqlite3'
    with sqlite3.connect(core.store.path) as source, sqlite3.connect(destination) as target: source.backup(target)
    restored = Core(Store(destination), enable_fixtures=True)
    try:
        if after_delete:
            assert restored.data_lifecycle.read('alice', run['case_id'])['local_content_removed']
            with pytest.raises(GateError, match='CASE_DELETED'): restored.input_timeline('alice', run['case_id'])
        else:
            # Old backups still contain content. No false erasure promise; a new instance cannot send it.
            assert restored.input_timeline('alice', run['case_id'])['snapshot_id'] == run['snapshot_id']
            assert not restored.data_permissions.read('alice', run['case_id'])['permissions']['report']
            with pytest.raises(GateError, match='DATA_PERMISSION_REQUIRED'):
                restored.data_permissions.authorize_run('alice', run['id'], 'report')
    finally: restored.close()


@pytest.mark.parametrize('termination', ['success', 'error', 'cancel', 'delete'])
def test_sdk_directory_removed_on_each_exit(core, run, tmp_path, monkeypatch, termination):
    monkeypatch.setattr(lifecycle, 'SESSION_ROOT', tmp_path / 'sessions')
    job_id = sdk_job(core, run)
    path = None
    try:
        with core.data_lifecycle.session('alice', run['id'], job_id) as sid:
            path = lifecycle.SESSION_ROOT / sid
            (path / 'config').mkdir()
            (path / 'config' / 'synthetic.json').write_text('SYNTHETIC_SESSION_MARKER')
            if termination == 'error': raise RuntimeError('simulated')
            if termination == 'cancel': core.cancel('alice', run['id'])
            if termination == 'delete':
                assert delete(core, run)['state'] == 'SDK_CLEANUP_PENDING'
    except RuntimeError:
        assert termination == 'error'
    assert path and not path.exists()
    with core.store.tx() as db:
        assert db.execute('SELECT state FROM sdk_sessions').fetchone()[0] == 'REMOVED'
    if termination == 'delete': assert core.data_lifecycle.read('alice', run['case_id'])['state'] == 'LOCAL_CONTENT_REMOVED'


def test_cleanup_failure_pending_and_startup_retry(core, run, tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, 'SESSION_ROOT', tmp_path / 'sessions')
    job_id = sdk_job(core, run)
    original = lifecycle.shutil.rmtree
    def fail(path): raise PermissionError('sensitive path should not be logged')
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.shutil, 'rmtree', fail)
        with pytest.raises(GateError, match='SDK_CLEANUP_FAILED'):
            with core.data_lifecycle.session('alice', run['id'], job_id) as sid:
                (lifecycle.SESSION_ROOT / sid / 'marker').write_text('synthetic')
                delete(core, run)
    assert core.data_lifecycle.read('alice', run['case_id'])['sdk_pending'] == {'CLEANUP_FAILED': 1}
    assert lifecycle.shutil.rmtree is original
    restarted = Core(Store(core.store.path), enable_fixtures=True)
    try:
        restarted.recover_interrupted()
        assert restarted.data_lifecycle.recover() == {'removed': 1, 'failed': 0}
        assert restarted.data_lifecycle.read('alice', run['case_id'])['state'] == 'LOCAL_CONTENT_REMOVED'
    finally: restarted.close()


def test_crash_recovery_keeps_legacy_and_foreign_ledger_directories(core, run, tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, 'SESSION_ROOT', tmp_path / 'sessions')
    job_id = sdk_job(core, run)
    sid = uuid4().hex
    path = lifecycle.SESSION_ROOT / sid; path.mkdir(parents=True)
    (path / 'interrupted.json').write_text('synthetic')
    legacy = lifecycle.SESSION_ROOT / uuid4().hex; legacy.mkdir()
    (legacy / 'keep').write_text('unmapped')
    with core.store.tx() as db:
        db.execute('INSERT INTO sdk_sessions VALUES(?,?,?,?,?,?)',
            (sid, run['case_id'], run['id'], job_id, core.data_lifecycle.scope, 'ACTIVE'))
    backup = tmp_path / 'backup.sqlite3'
    with sqlite3.connect(core.store.path) as src, sqlite3.connect(backup) as dst: src.backup(dst)
    restored = Core(Store(backup), enable_fixtures=True)
    try: assert restored.data_lifecycle.recover() == {'removed': 0, 'failed': 0}
    finally: restored.close()
    assert path.exists()
    core.recover_interrupted()
    assert core.data_lifecycle.recover()['removed'] == 1
    assert not path.exists() and (legacy / 'keep').read_text() == 'unmapped'


def test_cleanup_does_not_follow_symlinks_or_delete_preexisting_directory(core, run, tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, 'SESSION_ROOT', tmp_path / 'sessions')
    job_id = sdk_job(core, run)
    target = tmp_path / 'other-case'; target.mkdir(); (target / 'keep').write_text('protected')
    with pytest.raises(GateError, match='SDK_CLEANUP_FAILED'):
        with core.data_lifecycle.session('alice', run['id'], job_id) as sid:
            path = lifecycle.SESSION_ROOT / sid
            path.rmdir(); path.symlink_to(target, target_is_directory=True)
    assert (target / 'keep').read_text() == 'protected'
    path.unlink()
    core.data_lifecycle.recover()
    chosen = uuid4()
    monkeypatch.setattr(lifecycle, 'uuid4', lambda: chosen)
    path = lifecycle.SESSION_ROOT / chosen.hex; path.mkdir(); (path / 'keep').write_text('prior')
    with pytest.raises(GateError, match='SDK_STORAGE_OWNERSHIP'):
        with core.data_lifecycle.session('alice', run['id'], job_id): pytest.fail('must not yield')
    assert (path / 'keep').read_text() == 'prior'
    with pytest.raises(GateError, match='SDK_CLEANUP_FAILED'): core.data_lifecycle.recover()
    assert (path / 'keep').exists()


@pytest.mark.parametrize('kind', ['report', 'review'])
def test_delete_after_staging_blocks_registration_and_cleans_sdk(tmp_path, snapshot, core, run, kind):
    if kind == 'report':
        def action(client, kwargs):
            assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
            assert delete(core, run)['state'] == 'SDK_CLEANUP_PENDING'
            assert client.post('/tools/inspect_report_status', json={}).json()['error'] == 'CASE_DELETED'
            return completed()
        agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
        try:
            with pytest.raises(GateError, match='CASE_DELETED'): agent.generate('alice', run['id'])
        finally: agent.runner.close()
    else:
        def action(client, output, kwargs):
            assert client.post('/tools/submit_review', json=output).json()['ok']
            delete(core, run)
            return completed()
        core, agent, run, _ = prepare(tmp_path, snapshot, runner=ReviewRunner(action))
        try:
            with pytest.raises(GateError, match='CASE_DELETED'): core.review('alice', run['id'], 'medical')
        finally: core.close()
    assert count(core, 'drafts') == count(core, 'reviews') == count(core, 'releases') == 0
    assert core.data_lifecycle.read('alice', run['case_id'])['state'] == 'LOCAL_CONTENT_REMOVED'


def test_cleanup_failure_rejects_successful_staged_report(core, run, tmp_path, monkeypatch):
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    with monkeypatch.context() as patch:
        def fail(path): raise PermissionError('fake')
        patch.setattr(lifecycle.shutil, 'rmtree', fail)
        with pytest.raises(GateError, match='SDK_CLEANUP_FAILED'): agent.generate('alice', run['id'])
    agent.runner.close()
    assert count(core, 'drafts') == 0
    assert core.data_lifecycle.read('alice', run['case_id'])['sdk_pending'] == {'CLEANUP_FAILED': 1}
    core.data_lifecycle.recover()


def test_desktop_delete_is_fixed_and_has_no_agent_tool(core, run, monkeypatch):
    calls = []
    monkeypatch.setattr(desktop_client, '_request', lambda *args: calls.append(args) or {})
    body = {'case_id': run['case_id'], 'snapshot_id': run['snapshot_id'], 'contract_version': 'engineering-lifecycle-v1'}
    desktop_client.call(8787, 'case-delete', body)
    desktop_client.call(8787, 'case-lifecycle', {'case_id': run['case_id']})
    assert calls[0][:3] == (8787, 'DELETE', '/cases/' + run['case_id'])
    assert json.loads(calls[0][3]) == {k: v for k, v in body.items() if k != 'case_id'}
    assert calls[1] == (8787, 'GET', '/cases/' + run['case_id'] + '/lifecycle', None)
    with pytest.raises(GateError): desktop_client.call(8787, 'case-delete', body | {'case_id': '../escape'})
    from medical_harness.sdk_worker import TOOL_PREFIX, permitted
    for profile in ('report', 'review'):
        assert not permitted(TOOL_PREFIX + 'delete_case', {}, profile=profile)


@pytest.mark.parametrize('kind', ['report', 'review'])
def test_actual_sdk_files_exist_then_are_removed(tmp_path, snapshot, core, run, monkeypatch, kind):
    import httpx
    from medical_harness.sdk_worker import TOOL_PREFIX
    from test_sdk import SSE, message_events
    calls, binding = [], {}
    def provider(req):
        calls.append(json.loads(req.content))
        if kind == 'report': steps = [('inspect_report_contract', {}), ('submit_report_proposal', VALID)]
        else: steps = [('inspect_review_packet', {}), ('submit_review', dict(binding, verdict='pass', issues=[]))]
        phase = len(calls) - 1
        content = ({'type': 'tool_use', 'id': 'retention-' + str(phase), 'name': TOOL_PREFIX + steps[phase][0], 'input': steps[phase][1]}
                   if phase < 2 else {'type': 'text', 'text': 'done'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    if kind == 'report': agent = setup_agent(core, run, tmp_path, provider=provider)
    else:
        core, agent, run, hashes = prepare(tmp_path, snapshot, provider=provider)
        binding.update(hashes)
    observed = []
    original = core.data_lifecycle._cleanup
    def inspect_then_cleanup(sid):
        path = lifecycle.SESSION_ROOT / sid
        observed.append({'session_id': sid, 'file_count': sum(p.is_file() for p in path.rglob('*')),
                         'config_exists': (path / 'config/.claude.json').is_file(), 'tmp_exists': (path / 'tmp').is_dir()})
        original(sid)
        assert not path.exists()
    monkeypatch.setattr(core.data_lifecycle, '_cleanup', inspect_then_cleanup)
    try:
        if kind == 'report': assert agent.generate('alice', run['id'])['state'] == 'DRAFT_READY'
        else: assert core.review('alice', run['id'], 'medical')['verdict'] == 'pass'
        assert len(calls) == 3 and len(observed) == 1
        assert observed[0]['file_count'] > 0 and observed[0]['config_exists'] and observed[0]['tmp_exists']
        assert core.data_lifecycle.read('alice', run['case_id'])['sdk_pending'] == {}
        (tmp_path / 'sdk-retention-proof.json').write_text(canonical({'kind': kind, 'actual_sdk_worker': True,
            'paid_calls': 0, 'upstream': 'offline_mock', 'observed_before_cleanup': observed, 'directory_removed': True}))
    finally:
        if kind == 'report': agent.runner.close()
        else: core.close()


def test_local_http_delete_running_task_reaps_sdk_without_late_results(core, snapshot, tmp_path):
    import socket
    import time
    from threading import Event, Thread
    import httpx
    import uvicorn
    from medical_harness.agent_tasks import AgentTasks
    from medical_harness.report_agent import ReportAgent
    from test_sdk import SSE, message_events
    entered, release = Event(), Event()
    def provider(req):
        entered.set()
        assert release.wait(10)
        return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'done'})))
    key = tmp_path / 'offline.key'; key.write_text('offline'); key.chmod(0o600)
    reporter = ReportAgent(core, key, tmp_path / 'report', transport=httpx.MockTransport(provider))
    tasks = AgentTasks(core, reporter)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(core, {'offline-token': 'alice'}, agent_tasks=tasks),
                                          access_log=False, log_level='error'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True); thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline: time.sleep(.02)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False,
                          headers={'Authorization': 'Bearer offline-token'}) as client:
            case = client.post('/cases', json=snapshot).json()
            allow_data(core, case)
            task = client.post('/agent/tasks', json=dict(case, idempotency_key=uuid4().hex)).json()
            assert entered.wait(10)
            response = client.request('DELETE', '/cases/' + case['case_id'], json={
                'snapshot_id': case['snapshot_id'], 'contract_version': 'engineering-lifecycle-v1'})
            assert response.status_code == 200 and response.json()['state'] == 'SDK_CLEANUP_PENDING'
            release.set()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                result = client.get('/agent/tasks/' + task['id']).json()
                if not result['active']: break
                time.sleep(.05)
            assert result['state'] == 'FAILED' and result['invalid_reason'] == 'CASE_DELETED'
            assert client.get('/cases/' + case['case_id'] + '/lifecycle').json()['state'] == 'LOCAL_CONTENT_REMOVED'
            assert client.get('/cases').json() == []
            assert count(core, 'snapshots') == count(core, 'drafts') == count(core, 'reviews') == count(core, 'releases') == 0
            assert reporter.evidence['workers'][-1]['kind'] == 'worker_stopped'
            with core.store.tx() as db:
                sessions = db.execute('SELECT id,state FROM sdk_sessions').fetchall()
            assert sessions and all(row['state'] == 'REMOVED' and not (lifecycle.SESSION_ROOT / row['id']).exists() for row in sessions)
            (tmp_path / 'http-deletion-proof.json').write_text(canonical({'actual_local_http': True,
                'actual_sdk_worker': True, 'paid_calls': 0, 'upstream': 'offline_mock', 'pending_then_removed': True,
                'worker_reaped': True, 'task_state': result['state'], 'late_results': 0}))
    finally:
        release.set(); server.should_exit = True; thread.join(timeout=3); tasks.close(); sock.close()
    assert not thread.is_alive()


def test_sdk_directory_creation_failure_blocks_until_recovery(core, run, tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(lifecycle, 'SESSION_ROOT', tmp_path / 'sessions')
    job_id = sdk_job(core, run)
    original = Path.mkdir
    def fail(path, *args, **kwargs):
        if path.parent == lifecycle.SESSION_ROOT: raise PermissionError('do not log private path')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'mkdir', fail)
        with pytest.raises(GateError, match='SDK_STORAGE_FAILED'):
            with core.data_lifecycle.session('alice', run['id'], job_id): pytest.fail('cannot start')
    assert core.data_lifecycle.read('alice', run['case_id'])['sdk_pending'] == {'CLEANUP_FAILED': 1}
    with pytest.raises(GateError, match='SDK_CLEANUP_REQUIRED'):
        core.start_run('alice', run['case_id'], 'eval', 'cannot-start')
    assert core.data_lifecycle.recover() == {'removed': 1, 'failed': 0}
