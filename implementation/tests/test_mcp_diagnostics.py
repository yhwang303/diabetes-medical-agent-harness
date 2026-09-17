"""P03a diagnostics: no user-case execution, no live-provider transport."""

import json
import sqlite3
import httpx
from threading import Event, Thread

from fastapi.testclient import TestClient
import pytest

from medical_harness import desktop_client, mcp_diagnostics as diagnostics
from medical_harness.api import create_app
from medical_harness.contracts import GateError
from medical_harness.sdk_worker import PREDICTION_OPERATIONS


def rows(core):
    with core.store.tx() as db:
        return {table: [tuple(r) for r in db.execute('SELECT * FROM ' + table)]
                for table in ('cases', 'snapshots', 'jobs', 'runs', 'artifacts', 'data_permissions', 'sdk_sessions')}


def test_read_only_status_and_restart_reset(core, run, monkeypatch):
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', lambda path: pytest.fail('read must not execute'))
    before = rows(core)
    probe = diagnostics.McpDiagnostics(core)
    result = probe.status('alice')
    assert result['last_check']['state'] == 'NOT_CHECKED'
    assert result['tools'] == list(PREDICTION_OPERATIONS)
    assert not any(result[k] for k in ('prediction_entry_enabled', 'real_prediction_configured', 'rl_entry_enabled', 'real_rl_configured', 'online_provider_checked'))
    assert result['rl_mcp_configured'] and result['rl_tools'] == ['request_rl', 'inspect_rl']
    assert rows(core) == before
    with core.store.tx() as db:
        assert not db.execute("SELECT 1 FROM events WHERE kind LIKE 'mcp_check_%'").fetchone()
    assert diagnostics.McpDiagnostics(core, prediction_enabled=True).status('alice')['prediction_entry_enabled']


def test_actual_sdk_mcp_fixture_check_isolated_and_cleaned(core, run, monkeypatch):
    before = rows(core)
    from medical_harness.prediction_agent import PredictionAgent
    original = PredictionAgent.run
    evidence = []
    def inspect(agent, owner, run_id):
        assert agent.core is not core
        assert agent.key_path.name == 'offline.key'
        assert isinstance(agent.transport, httpx.MockTransport)
        assert agent.timeout == 20
        result = original(agent, owner, run_id)
        with agent.core.store.tx() as db:
            counts = {t: db.execute('SELECT count(*) FROM ' + t).fetchone()[0]
                      for t in ('jobs', 'artifacts', 'drafts', 'reviews', 'releases')}
            assert counts == {'jobs': 2, 'artifacts': 1, 'drafts': 0, 'reviews': 0, 'releases': 0}
            starts = [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE kind='worker_started'")]
            assert len({r['pid'] for r in starts}) == 2
            assert result['evidence']['origin'] == 'fixture'
        evidence.append(agent.evidence)
        return result
    monkeypatch.setattr(PredictionAgent, 'run', inspect)
    probe = diagnostics.McpDiagnostics(core)
    result = probe.check('alice')
    assert result['last_check']['state'] == 'PASSED', result
    assert result['last_check']['duration_ms'] > 0
    assert result == probe.status('alice')
    assert probe.status('bob')['last_check']['state'] == 'NOT_CHECKED'
    assert diagnostics.McpDiagnostics(core).status('alice')['last_check']['state'] == 'NOT_CHECKED'
    assert rows(core) == before
    assert not list((core.store.path.parent / 'mcp-checks').iterdir())
    assert len(evidence[0]['provider']) == 3
    with core.store.tx() as db:
        kinds = [r[0] for r in db.execute("SELECT kind FROM events WHERE kind LIKE 'mcp_check_%'")]
        assert kinds == ['mcp_check_started', 'mcp_check_finished']


@pytest.mark.parametrize('failure', [GateError('MCP_CHECK_INCOMPLETE'), RuntimeError('secret raw detail')])
def test_failed_retry_clears_success_and_hides_raw_error(core, monkeypatch, failure):
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', lambda path: None)
    probe = diagnostics.McpDiagnostics(core)
    assert probe.check('alice')['last_check']['state'] == 'PASSED'
    def fail(path): raise failure
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', fail)
    result = probe.check('alice')
    assert result['last_check']['state'] == 'FAILED'
    assert 'secret raw detail' not in json.dumps(result)
    assert not list((core.store.path.parent / 'mcp-checks').iterdir())


@pytest.mark.parametrize('installed', [None, '0.0.0'])
def test_missing_or_wrong_sdk_never_executes(core, monkeypatch, installed):
    def get_version(name):
        if installed is None: raise diagnostics.PackageNotFoundError(name)
        return installed
    monkeypatch.setattr(diagnostics, 'version', get_version)
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', lambda path: pytest.fail('must not execute'))
    result = diagnostics.McpDiagnostics(core).check('alice')
    assert result['last_check']['error'] == 'MCP_CHECK_SDK_VERSION'
    assert result['last_check']['state'] == 'FAILED'


def test_single_manual_check_and_running_status(core, monkeypatch):
    started, release = Event(), Event()
    def wait(path):
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', wait)
    probe = diagnostics.McpDiagnostics(core)
    thread = Thread(target=probe.check, args=('alice',)); thread.start()
    try:
        assert started.wait(3)
        assert probe.status('alice')['last_check']['state'] == 'RUNNING'
        for owner in ('alice', 'bob'):
            with pytest.raises(GateError, match='MCP_CHECK_BUSY'): probe.check(owner)
        assert probe.status('bob')['last_check']['state'] == 'NOT_CHECKED'
    finally:
        release.set(); thread.join(5)
    assert not thread.is_alive()
    assert probe.status('alice')['last_check']['state'] == 'PASSED'


def test_audit_failure_cannot_report_success(core, monkeypatch):
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', lambda path: None)
    original = core._event
    def event(db, run_id, kind, **details):
        if kind == 'mcp_check_finished': raise sqlite3.OperationalError('private database path')
        return original(db, run_id, kind, **details)
    monkeypatch.setattr(core, '_event', event)
    probe = diagnostics.McpDiagnostics(core)
    with pytest.raises(sqlite3.Error): probe.check('alice')
    assert probe.status('alice')['last_check']['state'] == 'FAILED'
    assert probe.status('alice')['last_check']['error'] == 'MCP_CHECK_AUDIT_FAILED'


def test_api_auth_origin_body_and_offline_result(core, monkeypatch):
    calls = []
    monkeypatch.setattr(diagnostics, 'offline_prediction_check', lambda path: calls.append(path))
    app = create_app(core, {'alice-token': 'alice'})
    with TestClient(app) as client:
        assert client.get('/mcp/status').status_code == 401
        assert client.post('/mcp/check').status_code == 401
        client.headers['Authorization'] = 'Bearer alice-token'
        assert client.post('/mcp/check', headers={'Origin': 'null'}).status_code == 403
        for body in ({}, {'case_id': 'private-case'}, {'key': 'private-key'}, {'url': 'https://example.invalid'}):
            response = client.post('/mcp/check', json=body)
            assert response.status_code == 422
            assert response.json() == {'error': 'CLIENT_BODY_FORBIDDEN'}
        assert not calls
        assert client.get('/mcp/status').json()['last_check']['state'] == 'NOT_CHECKED'
        response = client.post('/mcp/check')
        assert response.json()['last_check']['state'] == 'PASSED'
        assert response.headers['Cache-Control'] == 'no-store'
        assert len(calls) == 1
        assert not response.json()['prediction_entry_enabled']


@pytest.mark.parametrize('operation,method,path', [('mcp-status', 'GET', '/mcp/status'), ('mcp-check', 'POST', '/mcp/check')])
def test_desktop_fixed_operations(monkeypatch, operation, method, path):
    calls = []
    monkeypatch.setattr(desktop_client, '_request', lambda *args: calls.append(args) or {'ok': True})
    assert desktop_client.call(12345, operation, {}) == {'ok': True}
    assert calls == [(12345, method, path, None)]
    with pytest.raises(GateError): desktop_client.call(12345, operation, {'case_id': 'a' * 32})
    assert len(calls) == 1
