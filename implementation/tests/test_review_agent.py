import json
from threading import Event

import httpx
import pytest
from conftest import allow_data

from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.review_agent import ReviewAgent
from medical_harness.sdk_worker import REVIEW_EXECUTORS, REVIEW_OPERATIONS, TOOL_PREFIX, permitted
from medical_harness.store import Store
from medical_harness.workers import group_rss
from test_sdk import SSE, message_events, request


def prepare(tmp_path, snapshot, *, provider=None, runner=None):
    key = tmp_path / 'offline.key'
    key.write_text('offline-secret')
    key.chmod(0o600)
    transport = httpx.MockTransport(provider or (lambda req: httpx.Response(200,
        stream=SSE(message_events({'type': 'text', 'text': 'done'})))))
    reviewer = ReviewAgent(key, tmp_path / 'review', transport=transport, runner=runner)
    core = Core(Store(tmp_path / 'ledger.sqlite3'), enable_fixtures=True, review_agent=reviewer)
    case = core.create_case('alice', snapshot)
    for purpose in ('medical_review', 'ethics_review'): allow_data(core, case, purpose)
    run = core.start_run('alice', case['case_id'], 'eval', 'review-test')
    core.execute_models('alice', run['id'])
    job = core.create_draft_job('alice', run['id'])
    draft = core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
    return core, reviewer, run, {'report_hash': draft['report_hash'], 'evidence_hash': job['evidence_hash']}


def count(core, table):
    with core.store.tx() as db:
        return db.execute('SELECT count(*) FROM ' + table).fetchone()[0]


@pytest.mark.parametrize('verdict', ['pass', 'fail', 'abstain'])
def test_two_actual_sdk_reviewers_register_real_origin_and_enforce_verdict(tmp_path, snapshot, verdict):
    calls = []
    binding = {}
    def provider(req):
        body = json.loads(req.content)
        calls.append(body)
        assert {t['name'] for t in body['tools']} == {TOOL_PREFIX + x for x in REVIEW_OPERATIONS}
        phase = (len(calls)-1) % 3
        content = ({'type': 'tool_use', 'id': 'inspect-' + str(len(calls)), 'name': TOOL_PREFIX + 'inspect_review_packet', 'input': {}}
            if phase == 0 else {'type': 'tool_use', 'id': 'submit-' + str(len(calls)), 'name': TOOL_PREFIX + 'submit_review',
                'input': dict(binding, verdict=verdict, issues=[] if verdict == 'pass' else ['review_uncertain' if verdict == 'abstain' else 'unsupported_claim'])}
            if phase == 1 else {'type': 'text', 'text': 'arbitrary final prose is not a verdict'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    core, reviewer, run, hashes = prepare(tmp_path, snapshot, provider=provider)
    binding.update(hashes)
    try:
        medical = core.review('alice', run['id'], 'medical')
        assert medical['verdict'] == verdict
        with pytest.raises(GateError, match='REVIEW_REQUIRED'):
            core.release('alice', run['id'])
        ethics = core.review('alice', run['id'], 'ethics')
        assert ethics['state'] == ('REVIEWED' if verdict == 'pass' else 'REVIEW_BLOCKED')
        assert count(core, 'reviews') == 2 and count(core, 'releases') == 0
        if verdict != 'pass':
            with pytest.raises(GateError, match='REVIEW_REQUIRED'):
                core.release('alice', run['id'])
        assert len(calls) == 6
        sessions = [x['sdk']['results'][-1]['session_id'] for x in reviewer.evidence]
        assert len(set(sessions)) == 2
        assert calls[0]['system'] != calls[3]['system']
        assert all(x['workers'][-1]['exit_code'] == 0 for x in reviewer.evidence)
        with core.store.tx() as db:
            records = [json.loads(r[0]) for r in db.execute('SELECT body FROM reviews')]
            assert {r['role'] for r in records} == {'medical', 'ethics'}
            assert all(r['origin'] == 'model' and r['report_hash'] == binding['report_hash'] for r in records)
            assert all(r['producer'] == REVIEW_EXECUTORS[r['role']].identity for r in records)
        assert 'arbitrary final prose' not in json.dumps(reviewer.evidence)
    finally:
        core.close()


class FaultRunner:
    def __init__(self, action, *, inspect=True):
        self.action, self.inspect = action, inspect

    def run(self, executor, envelope, **kwargs):
        with httpx.Client(base_url=f"http://127.0.0.1:{envelope['port']}", trust_env=False,
                          headers={'Authorization': 'Bearer ' + envelope['capability']}) as client:
            assert client.post('/v1/messages', json=request()).status_code == 200
            if not self.inspect:
                return self.action(client, {}, kwargs)
            packet = client.post('/tools/inspect_review_packet', json={}).json()
            assert packet['ok']
            assert packet['role'] == envelope['scenario'].removeprefix('review_')
            assert 'reviews' not in packet and 'capability' not in packet
            output = {k: packet['packet'][k] for k in ('report_hash', 'evidence_hash')}
            output.update(verdict='pass', issues=[])
            return self.action(client, output, kwargs)

    def close(self):
        pass


def completed():
    return {'results': [{'is_error': False, 'terminal_reason': 'completed', 'session_id': 'offline'}]}


@pytest.mark.parametrize('mutation,code', [({'report_hash': 'wrong'}, 'REVIEW_BINDING'),
    ({'evidence_hash': 'wrong'}, 'REVIEW_BINDING'), ({'role': 'ethics'}, 'INVALID_REVIEW_OUTPUT'),
    ({'producer': 'trusted'}, 'INVALID_REVIEW_OUTPUT'), ({'dose': 999}, 'INVALID_REVIEW_OUTPUT'),
    ({'verdict': 'pass', 'issues': ['unsafe_action']}, 'INVALID_REVIEW_OUTPUT'),
    ({'verdict': 'abstain', 'issues': []}, 'REVIEW_REASON_REQUIRED'),
    ({'verdict': 'fail', 'issues': []}, 'REVIEW_REASON_REQUIRED')])
def test_forged_or_inconsistent_review_is_not_registered(tmp_path, snapshot, mutation, code):
    def action(client, output, kwargs):
        result = client.post('/tools/submit_review', json=output | mutation).json()
        assert result['ok'] is False and result['error'] == code
        return completed()
    core, reviewer, run, _ = prepare(tmp_path, snapshot, runner=FaultRunner(action))
    try:
        with pytest.raises(GateError, match='REVIEW_OUTPUT_MISSING'):
            core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == 0
        assert core.status('alice', run['id'])['state'] == 'REVIEW_BLOCKED'
    finally:
        core.close()


@pytest.mark.parametrize('fault,expected', [('timeout', 'EXECUTOR_TIMEOUT'), ('crash', 'REVIEW_AGENT_FAILED'),
    ('incomplete', 'REVIEW_AGENT_INCOMPLETE'), ('cancel', 'CANCELLED'), ('replace', 'STALE_ATTEMPT')])
def test_staged_review_is_discarded_after_failure(tmp_path, snapshot, fault, expected):
    def action(client, output, kwargs):
        assert client.post('/tools/submit_review', json=output).json()['ok']
        assert count(core, 'reviews') == 0
        if fault == 'timeout':
            raise GateError('EXECUTOR_TIMEOUT')
        if fault == 'crash':
            raise RuntimeError('secret provider text')
        if fault == 'incomplete':
            return {'results': [{'is_error': True, 'terminal_reason': 'completed'}]}
        if fault == 'cancel':
            core.cancel('alice', run['id'])
        if fault == 'replace':
            core.create_draft_job('alice', run['id'])
        return completed()
    core, reviewer, run, _ = prepare(tmp_path, snapshot, runner=FaultRunner(action))
    try:
        with pytest.raises(GateError, match=expected):
            core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == 0
        assert 'secret provider text' not in json.dumps(reviewer.evidence)
    finally:
        core.close()


def test_real_reviewer_configuration_is_frozen_and_never_falls_back(tmp_path, snapshot):
    core, reviewer, run, _ = prepare(tmp_path, snapshot)
    try:
        with core.store.tx() as db:
            versions = json.loads(db.execute('SELECT versions FROM runs').fetchone()[0])
            assert all(x.version in versions for x in REVIEW_EXECUTORS.values())
            assert 'fixture-medical-v1' not in versions and 'fixture-ethics-v1' not in versions
        restarted = Core(core.store, enable_fixtures=True)
        try:
            with pytest.raises(GateError, match='REVIEWER_NOT_CONFIGURED'):
                restarted.review('alice', run['id'], 'medical')
            assert count(core, 'reviews') == 0
        finally:
            restarted.close()
    finally:
        core.close()


def test_review_profile_disallows_other_roles_reports_and_system_tools():
    for name in ('Bash', 'Agent', 'Task', TOOL_PREFIX + 'release_probe', TOOL_PREFIX + 'submit_report_proposal'):
        assert not permitted(name, {}, profile='review')
    assert not permitted(TOOL_PREFIX + 'inspect_review_packet', {'run_id': 'other'}, profile='review')
    assert not permitted(TOOL_PREFIX + 'inspect_review_packet', {}, profile='review', child=True)
    assert permitted(TOOL_PREFIX + 'inspect_review_packet', {}, profile='review')


def test_audit_failure_prevents_review_registration(tmp_path, snapshot, monkeypatch):
    def action(client, output, kwargs):
        assert client.post('/tools/submit_review', json=output).status_code == 500
        return completed()
    core, reviewer, run, _ = prepare(tmp_path, snapshot, runner=FaultRunner(action))
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == 'review_agent_tool' and metadata.get('tool') == 'submit_review':
            raise RuntimeError('audit failure')
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, '_event', fail)
    try:
        with pytest.raises(GateError, match='REVIEW_OUTPUT_MISSING'):
            core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == 0
    finally:
        core.close()


def test_read_required_and_verdict_submission_is_single_use(tmp_path, snapshot):
    binding = {}
    def action(client, output, kwargs):
        output = dict(binding, verdict='pass', issues=[])
        assert client.post('/tools/submit_review', json=output).json()['error'] == 'REVIEW_PACKET_REQUIRED'
        assert client.post('/tools/inspect_review_packet', json={'role': 'ethics'}).json()['error'] == 'INVALID_TOOL_ARGUMENTS'
        assert client.post('/tools/inspect_review_packet', json={}).json()['ok']
        assert client.post('/tools/submit_review', json=output).json()['ok']
        assert client.post('/tools/submit_review', json=output).json()['error'] == 'REVIEW_ALREADY_STAGED'
        return completed()
    core, reviewer, run, hashes = prepare(tmp_path, snapshot, runner=FaultRunner(action, inspect=False))
    binding.update(hashes)
    try:
        assert core.review('alice', run['id'], 'medical')['verdict'] == 'pass'
        assert count(core, 'reviews') == 1
    finally:
        core.close()


def test_old_fixture_run_keeps_frozen_reviewer_config(tmp_path, snapshot):
    store = Store(tmp_path / 'old.sqlite3')
    old = Core(store, enable_fixtures=True)
    case = old.create_case('alice', snapshot)
    run = old.start_run('alice', case['case_id'], 'eval', 'old')
    old.execute_models('alice', run['id'])
    job = old.create_draft_job('alice', run['id'])
    old.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
    old.close()
    reviewer = ReviewAgent(tmp_path / 'unused.key', tmp_path / 'review', runner=FaultRunner(lambda *args: pytest.fail('old fixture run cannot switch to SDK')))
    current = Core(store, enable_fixtures=True, review_agent=reviewer)
    try:
        assert current.review('alice', run['id'], 'medical')['verdict'] == 'pass'
        with store.tx() as db:
            assert json.loads(db.execute('SELECT body FROM reviews').fetchone()[0])['origin'] == 'fixture'
        assert not (tmp_path / 'review/budget.json').exists()
    finally:
        current.close()


def test_one_real_pass_cannot_cover_other_reviewer_failure(tmp_path, snapshot):
    calls = []
    def action(client, output, kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise GateError('EXECUTOR_TIMEOUT')
        assert client.post('/tools/submit_review', json=output).json()['ok']
        return completed()
    core, reviewer, run, _ = prepare(tmp_path, snapshot, runner=FaultRunner(action))
    try:
        assert core.review('alice', run['id'], 'medical')['verdict'] == 'pass'
        with pytest.raises(GateError, match='EXECUTOR_TIMEOUT'):
            core.review('alice', run['id'], 'ethics')
        assert count(core, 'reviews') == 1 and core.status('alice', run['id'])['state'] == 'REVIEW_BLOCKED'
        with pytest.raises(GateError, match='REVIEW_REQUIRED'):
            core.release('alice', run['id'])
        core.review('alice', run['id'], 'medical')  # Cached review does not cause another paid call.
        assert len(calls) == 2
    finally:
        core.close()


def test_actual_review_sdk_cancel_after_staging_reaps_group(tmp_path, snapshot):
    calls, binding, stop = [], {}, Event()
    class Stalled(httpx.SyncByteStream):
        def __iter__(self):
            while not stop.wait(0.05):
                pass
            yield b''
    def provider(req):
        calls.append(True)
        if len(calls) == 3:
            core.cancel('alice', run['id'])
            return httpx.Response(200, stream=Stalled())
        content = ({'type': 'tool_use', 'id': 'inspect', 'name': TOOL_PREFIX + 'inspect_review_packet', 'input': {}}
            if len(calls) == 1 else {'type': 'tool_use', 'id': 'submit', 'name': TOOL_PREFIX + 'submit_review',
                'input': dict(binding, verdict='pass', issues=[])})
        return httpx.Response(200, stream=SSE(message_events(content)))
    core, reviewer, run, hashes = prepare(tmp_path, snapshot, provider=provider)
    binding.update(hashes)
    try:
        with pytest.raises(GateError, match='CANCELLED'):
            core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == 0
        assert reviewer.evidence[0]['tools'][-1]['name'] == 'submit_review' and reviewer.evidence[0]['tools'][-1]['ok']
        events = reviewer.evidence[0]['workers']
        assert events[-1]['exit_code'] == -9
        assert all(group_rss(e['pid']) == 0 for e in events if e['kind'] == 'worker_started')
    finally:
        stop.set()
        core.close()
