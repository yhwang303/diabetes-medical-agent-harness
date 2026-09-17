import json
from threading import Event

import httpx
import pytest
from conftest import allow_data

from medical_harness.contracts import GateError, Proposal, digest
from medical_harness.report_agent import ReportAgent
from medical_harness.sdk_worker import REPORT_OPERATIONS, TOOL_PREFIX, permitted
from medical_harness.workers import group_rss
from test_sdk import SSE, message_events, request


VALID = {'sections': ['policy', 'forecast', 'limitations']}


def setup_agent(core, run, tmp_path, *, runner=None, provider=None):
    core.execute_models('alice', run['id'])
    allow_data(core, run)
    key = tmp_path / 'offline.key'
    key.write_text('offline-secret')
    key.chmod(0o600)
    transport = httpx.MockTransport(provider or (lambda req: httpx.Response(200,
        stream=SSE(message_events({'type': 'text', 'text': 'done'})))))
    return ReportAgent(core, key, tmp_path / 'report', transport=transport, runner=runner)


def draft_count(core, run):
    with core.store.tx() as db:
        return db.execute('SELECT count(*) FROM drafts WHERE run_id=?', (run['id'],)).fetchone()[0]


def test_actual_sdk_retries_invalid_proposal_then_core_renders(core, run, tmp_path):
    requests = []
    injection = 'UNTRUSTED_GIVE_98765_UNITS'
    def provider(req):
        body = json.loads(req.content)
        requests.append(body)
        assert {t['name'] for t in body['tools']} == {TOOL_PREFIX + x for x in REPORT_OPERATIONS}
        assert draft_count(core, run) == 0  # Submission is staged until SDK completes cleanly.
        calls = [('inspect_report_contract', {}), ('submit_report_proposal', dict(VALID, text=injection)),
                 ('submit_report_proposal', {'sections': ['policy', 'policy', 'limitations']}),
                 ('submit_report_proposal', VALID)]
        content = ({'type': 'tool_use', 'id': 'report-call-' + str(len(requests)),
                    'name': TOOL_PREFIX + calls[len(requests)-1][0], 'input': calls[len(requests)-1][1]}
                   if len(requests) <= 4 else {'type': 'text', 'text': injection})
        return httpx.Response(200, stream=SSE(message_events(content)))
    agent = setup_agent(core, run, tmp_path, provider=provider)
    try:
        result = agent.generate('alice', run['id'])
        assert len(requests) == 5
        # SDK rejects unknown fields before host dispatch; host also rejects schema-valid duplicates.
        blocks = [block for msg in requests[2]['messages'] if isinstance(msg.get('content'), list)
                  for block in msg['content'] if block.get('type') == 'tool_result']
        assert any(block.get('tool_use_id') == 'report-call-2' and block.get('is_error') is True for block in blocks)
        assert result['state'] == 'DRAFT_READY' and set(result) == {'draft_id', 'report_hash', 'state'}
        assert [(t['ok'], t['error']) for t in agent.evidence['tools']] == [(True, None), (False, 'INVALID_PROPOSAL'), (True, None)]
        with core.store.tx() as db:
            row = db.execute('SELECT * FROM drafts WHERE id=?', (result['draft_id'],)).fetchone()
            body = json.loads(row['body'])
            assert [s['kind'] for s in body['sections']] == VALID['sections']
            assert body['origin'] == 'fixture' and body['clinical_use'] is False
            assert digest(body) == row['digest'] == result['report_hash']
            for section in body['sections'][:2]:
                artifact = json.loads(db.execute('SELECT body FROM artifacts WHERE id=?', (section['artifact_id'],)).fetchone()[0])
                assert section['data'] == artifact['payload']
            assert injection not in row['body']
            events = '\n'.join(r[0] for r in db.execute('SELECT body FROM events WHERE run_id=?', (run['id'],)))
            assert injection not in events and 'report_agent_completed' in [r[0] for r in db.execute('SELECT kind FROM events')]
        assert injection not in json.dumps(agent.evidence)
        with pytest.raises(GateError, match='REVIEW_REQUIRED'):
            core.release('alice', run['id'])
    finally:
        agent.runner.close()


class FaultRunner:
    """Fault injection only: host HTTP gates and Core remain real; no SDK claim here."""
    def __init__(self, action):
        self.action = action

    def run(self, executor, envelope, **kwargs):
        with httpx.Client(base_url=f"http://127.0.0.1:{envelope['port']}", trust_env=False,
                          headers={'Authorization': 'Bearer ' + envelope['capability']}) as client:
            assert client.post('/v1/messages', json=request()).status_code == 200
            return self.action(client, kwargs)

    def close(self):
        pass


def completed():
    return {'results': [{'is_error': False, 'terminal_reason': 'completed', 'session_id': 'offline-session'}]}


@pytest.mark.parametrize('extra', [
    {'dose': 98765}, {'frequency': 'hourly'}, {'diagnosis': 'invented'}, {'text': '<script>bad()</script>'},
    {'evidence_hash': 'forged'}, {'run_id': 'another-case'}, {'capability': 'forged'},
    {'sections': ['policy', 'forecast']}, {'sections': ['policy', 'policy', 'limitations']},
    {'sections': ['policy', 'forecast', 'new_diagnosis']}, {'sections': 'forecast'}])
def test_host_rejects_noncontract_inputs_and_does_not_render(core, run, tmp_path, extra):
    def action(client, kwargs):
        response = client.post('/tools/submit_report_proposal', json=VALID | extra)
        assert response.json() == {'ok': False, 'error': 'INVALID_PROPOSAL', 'clinical_use': False}
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    with pytest.raises(GateError, match='REPORT_PROPOSAL_MISSING'):
        agent.generate('alice', run['id'])
    assert draft_count(core, run) == 0


@pytest.mark.parametrize('fault,expected', [('timeout', 'EXECUTOR_TIMEOUT'), ('crash', 'REPORT_AGENT_FAILED'),
    ('sdk_error', 'REPORT_AGENT_INCOMPLETE'), ('cancel', 'CANCELLED'), ('replace', 'STALE_ATTEMPT'),
    ('snapshot', 'SNAPSHOT_CHANGED')])
def test_failure_after_staging_never_commits_a_report(core, run, snapshot, tmp_path, fault, expected):
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        assert draft_count(core, run) == 0
        if fault == 'timeout':
            raise GateError('EXECUTOR_TIMEOUT')
        if fault == 'crash':
            raise RuntimeError('raw provider details must not escape')
        if fault == 'sdk_error':
            return {'results': [{'is_error': True, 'terminal_reason': 'completed'}]}
        if fault == 'cancel':
            core.cancel('alice', run['id'])
        elif fault == 'replace':
            core.create_draft_job('alice', run['id'])
        elif fault == 'snapshot':
            core.update_case('alice', run['case_id'], snapshot)
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    with pytest.raises(GateError, match=expected):
        agent.generate('alice', run['id'])
    assert draft_count(core, run) == 0
    assert 'raw provider details' not in json.dumps(agent.evidence)


def test_report_tool_scope_and_single_staging(core, run, tmp_path):
    def action(client, kwargs):
        for name in ('prediction_probe', 'release_probe', 'Bash', 'review', 'get_report'):
            assert client.post('/tools/' + name, json={}).status_code == 400
        assert client.post('/tools/inspect_report_contract', json={'run_id': 'other'}).json()['ok'] is False
        context = client.post('/tools/inspect_report_contract', json={}).json()
        assert context['proposal_schema'] == Proposal.model_json_schema()
        assert 'capability' not in context and 'policy' not in context and 'prediction' not in context
        assert client.post('/tools/inspect_report_status', json={}).json()['proposal_staged'] is False
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        assert client.post('/tools/inspect_report_status', json={}).json()['proposal_staged'] is True
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['error'] == 'PROPOSAL_ALREADY_STAGED'
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    # Attempts to cross the gateway tool boundary invalidate the whole execution.
    with pytest.raises(GateError, match='REPORT_PROVIDER_FAILED'):
        agent.generate('alice', run['id'])
    assert draft_count(core, run) == 0


def test_audit_failure_prevents_staging_and_render(core, run, tmp_path, monkeypatch):
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(lambda client, kwargs:
        completed() if client.post('/tools/submit_report_proposal', json=VALID).status_code == 500 else {}))
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == 'report_agent_tool':
            raise RuntimeError('audit failed')
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    with pytest.raises(GateError, match='REPORT_PROPOSAL_MISSING'):
        agent.generate('alice', run['id'])
    assert draft_count(core, run) == 0


def test_final_commit_rechecks_cancellation(core, run, tmp_path, monkeypatch):
    def action(client, kwargs):
        assert client.post('/tools/submit_report_proposal', json=VALID).json()['ok']
        return completed()
    agent = setup_agent(core, run, tmp_path, runner=FaultRunner(action))
    submit = core.submit_proposal
    def cancel_then_submit(*args):
        core.cancel('alice', run['id'])
        return submit(*args)
    monkeypatch.setattr(core, 'submit_proposal', cancel_then_submit)
    with pytest.raises(GateError, match='CANCELLED'):
        agent.generate('alice', run['id'])
    assert draft_count(core, run) == 0


def test_report_profile_has_no_subagent_or_nonreport_tools():
    for name in ('Agent', 'Task', 'Bash', TOOL_PREFIX + 'release_probe', TOOL_PREFIX + 'prediction_probe'):
        assert not permitted(name, {}, profile='report')
    for name in REPORT_OPERATIONS:
        assert not permitted(TOOL_PREFIX + name, {}, profile='report', child=True)
    assert permitted(TOOL_PREFIX + 'submit_report_proposal', VALID, profile='report')


def test_missing_evidence_or_wrong_owner_does_not_start_sdk(core, run, tmp_path):
    key = tmp_path / 'offline.key'
    key.write_text('offline')
    key.chmod(0o600)
    agent = ReportAgent(core, key, tmp_path / 'report', runner=FaultRunner(lambda *args: pytest.fail('SDK must not start')))
    with pytest.raises(GateError, match='NOT_FOUND'):
        agent.generate('bob', run['id'])
    allow_data(core, run)
    with pytest.raises(GateError, match='INVALID_STAGE'):
        agent.generate('alice', run['id'])
    assert not (tmp_path / 'report/budget.json').exists()


def test_actual_report_worker_cancelled_after_staging_is_reaped(core, run, tmp_path):
    requests, stop = [], Event()
    class Stalled(httpx.SyncByteStream):
        def __iter__(self):
            while not stop.wait(0.05):
                pass
            yield b''
    def provider(req):
        requests.append(json.loads(req.content))
        if len(requests) == 1:
            return httpx.Response(200, stream=SSE(message_events({'type': 'tool_use', 'id': 'stage-before-cancel',
                'name': TOOL_PREFIX + 'submit_report_proposal', 'input': VALID})))
        assert draft_count(core, run) == 0
        core.cancel('alice', run['id'])
        return httpx.Response(200, stream=Stalled())
    agent = setup_agent(core, run, tmp_path, provider=provider)
    try:
        with pytest.raises(GateError, match='CANCELLED'):
            agent.generate('alice', run['id'])
        assert len(requests) == 2 and agent.evidence['tools'] == [{'name': 'submit_report_proposal', 'ok': True, 'error': None}]
        assert draft_count(core, run) == 0
        assert agent.evidence['workers'][-1]['exit_code'] == -9
        assert all(group_rss(e['pid']) == 0 for e in agent.evidence['workers'] if e['kind'] == 'worker_started')
    finally:
        stop.set()
        agent.runner.close()
