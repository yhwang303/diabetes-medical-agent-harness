"""Real reviewer path with offline fault injection; paid evidence is verified separately."""

import json
import sqlite3
from pathlib import Path
from threading import Event

import httpx
import pytest

from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core
from medical_harness.review_agent import ReviewAgent
from medical_harness.store import Store
from medical_harness.sdk_worker import REVIEW_EXECUTORS, REVIEW_PROMPTS, TOOL_PREFIX
from medical_harness.workers import group_rss
from test_review_agent import FaultRunner, completed, count, prepare
from test_sdk import SSE, message_events


def reviewed(tmp_path, snapshot):
    def action(client, output, kwargs):
        assert client.post('/tools/submit_review', json=output).json()['ok']
        return completed()
    core, reviewer, run, hashes = prepare(tmp_path, snapshot, runner=FaultRunner(action))
    for role in ('medical', 'ethics'):
        core.review('alice', run['id'], role)
    return core, reviewer, run, hashes


def test_cached_review_must_revalidate_full_record(tmp_path, snapshot):
    core, reviewer, run, _ = reviewed(tmp_path, snapshot)
    try:
        with core.store.tx() as db:
            row = db.execute("SELECT id,body FROM reviews WHERE role='medical'").fetchone()
            body = json.loads(row['body'])
            body['producer'] = 'forged'
            db.execute('UPDATE reviews SET body=? WHERE id=?', (canonical(body), row['id']))
        with pytest.raises(GateError, match='REVIEW_INTEGRITY|REVIEW_INVALID'):
            core.review('alice', run['id'], 'medical')
    finally:
        core.close()


def test_prompt_change_without_version_bump_invalidates_review(tmp_path, snapshot, monkeypatch):
    core, reviewer, run, _ = reviewed(tmp_path, snapshot)
    try:
        monkeypatch.setitem(REVIEW_PROMPTS, 'medical', REVIEW_PROMPTS['medical'] + ' Changed rubric.')
        with pytest.raises(GateError, match='REVIEW_CONFIG_CHANGED'):
            core.release('alice', run['id'])
        assert count(core, 'releases') == 0
    finally:
        core.close()


def test_review_job_input_digest_rechecked_at_release(tmp_path, snapshot):
    core, reviewer, run, _ = reviewed(tmp_path, snapshot)
    try:
        with core.store.tx() as db:
            db.execute("UPDATE jobs SET input_digest='wrong' WHERE step='review_medical'")
        with pytest.raises(GateError, match='REVIEW_INVALID'):
            core.release('alice', run['id'])
        assert count(core, 'releases') == 0
    finally:
        core.close()


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp('bound-review-baseline')
    snapshot = json.loads((Path(__file__).parents[1] / 'examples/synthetic-case.json').read_text())
    core, reviewer, run, hashes = reviewed(path, snapshot)
    core.close()
    return core.store.path


@pytest.fixture
def bound(baseline, tmp_path):
    path = tmp_path / 'copy.sqlite3'
    with sqlite3.connect(baseline) as source, sqlite3.connect(path) as target:
        source.backup(target)
    reviewer = ReviewAgent(tmp_path / 'unused.key', tmp_path / 'unused',
                           runner=FaultRunner(lambda *args: pytest.fail('no new model call expected')))
    core = Core(Store(path), enable_fixtures=True, review_agent=reviewer)
    with core.store.tx() as db:
        run = dict(db.execute('SELECT * FROM runs').fetchone())
    yield core, run
    core.close()


@pytest.mark.parametrize('field', ['system', 'task', 'tool', 'model', 'sdk', 'cli', 'endpoint', 'output_cap'])
def test_configuration_content_change_blocks_cache_release_and_read(bound, monkeypatch, field):
    import medical_harness.sdk_worker as sdk
    import medical_harness.flash_gateway as gateway
    core, run = bound
    released = core.release('alice', run['id'])
    if field == 'system':
        monkeypatch.setitem(sdk.REVIEW_PROMPTS, 'medical', sdk.REVIEW_PROMPTS['medical'] + ' changed')
    elif field == 'task':
        monkeypatch.setitem(sdk.REVIEW_TASKS, 'medical', sdk.REVIEW_TASKS['medical'] + ' changed')
    elif field == 'tool':
        monkeypatch.setitem(sdk.REVIEW_TOOL_DESCRIPTIONS, 'submit_review', 'different semantics')
    else:
        module, name, value = {'model': (sdk, 'MODEL', 'other-flash'), 'sdk': (sdk, 'SDK_VERSION', 'changed'),
            'cli': (sdk, 'CLI_VERSION', 'changed'), 'endpoint': (gateway, 'UPSTREAM', 'https://invalid.example/messages'),
            'output_cap': (gateway, 'MAX_OUTPUT_TOKENS', 385)}[field]
        monkeypatch.setattr(module, name, value)
    for action in (lambda: core.review('alice', run['id'], 'medical'), lambda: core.release('alice', run['id']),
                   lambda: core.get_release('alice', released['release_id'])):
        with pytest.raises(GateError, match='REVIEW_CONFIG_CHANGED'):
            action()
    assert core.status('alice', run['id'])['currently_valid'] is False


@pytest.mark.parametrize('field', ['draft_id', 'revision', 'role', 'config_hash', 'case_id', 'snapshot_id', 'run_id'])
def test_binding_fields_cannot_be_changed_even_with_recomputed_record_hash(bound, field):
    core, run = bound
    with core.store.tx() as db:
        row = db.execute("SELECT * FROM reviews WHERE role='medical'").fetchone()
        body = json.loads(row['body'])
        body['binding'][field] = 999 if field == 'revision' else 'other'
        db.execute('UPDATE reviews SET body=?,digest=? WHERE id=?', (canonical(body), digest(body), row['id']))
    for action in (lambda: core.review('alice', run['id'], 'medical'), lambda: core.release('alice', run['id'])):
        with pytest.raises(GateError, match='REVIEW_INVALID'):
            action()
    assert count(core, 'releases') == 0


@pytest.mark.parametrize('action,reason', [('snapshot', 'SNAPSHOT_CHANGED'), ('reviewer', 'VERSION_REVOKED'),
    ('rules', 'VERSION_REVOKED'), ('template', 'VERSION_REVOKED'), ('cancel', 'CANCELLED'), ('expiry', 'RUN_EXPIRED')])
def test_real_review_invalidation_blocks_published_read(bound, snapshot, action, reason):
    from medical_harness.render import RULE_VERSION, TEMPLATE_VERSION
    core, run = bound
    released = core.release('alice', run['id'])
    if action == 'snapshot':
        core.update_case('alice', run['case_id'], snapshot)
    elif action in ('reviewer', 'rules', 'template'):
        core.revoke_version({'reviewer': REVIEW_EXECUTORS['medical'].version, 'rules': RULE_VERSION, 'template': TEMPLATE_VERSION}[action])
    elif action == 'cancel':
        core.cancel('alice', run['id'])
    else:
        core.clock = lambda: run['expires'] + 1
    with pytest.raises(GateError, match=reason):
        core.get_release('alice', released['release_id'])
    with pytest.raises(GateError, match=reason):
        core.review('alice', run['id'], 'medical')


def test_same_bytes_new_revision_still_requires_both_new_reviews(bound):
    core, run = bound
    with core.store.tx() as db:
        original = dict(db.execute('SELECT * FROM drafts').fetchone())
        old_review_ids = [r[0] for r in db.execute('SELECT id FROM reviews')]
    proposal = {'sections': [section['kind'] for section in json.loads(original['body'])['sections']]}
    job = core.create_draft_job('alice', run['id'])
    current = core.submit_proposal(job['job_id'], job['capability'], canonical(proposal))
    assert current['report_hash'] == original['digest'] and current['draft_id'] != original['id']
    with pytest.raises(GateError, match='REVIEW_REQUIRED'):
        core.release('alice', run['id'])
    with core.store.tx() as db:
        assert [r[0] for r in db.execute('SELECT id FROM reviews')] == old_review_ids
        assert db.execute('SELECT count(*) FROM reviews WHERE draft_id=?', (current['draft_id'],)).fetchone()[0] == 0


def test_restart_preserves_valid_bindings_and_blocks_missing_legacy_fingerprint(bound, tmp_path):
    core, run = bound
    before = core.review('alice', run['id'], 'medical')
    core.close()
    fresh = Core(Store(core.store.path), enable_fixtures=True, review_agent=ReviewAgent(tmp_path / 'unused', tmp_path / 'review'))
    try:
        assert fresh.review('alice', run['id'], 'medical') == before
        with fresh.store.tx() as db:
            old_rows = [tuple(r) for r in db.execute('SELECT * FROM reviews')]
            db.execute('DELETE FROM review_configs')  # Simulate pre-fingerprint real review history.
        with pytest.raises(GateError, match='REVIEW_CONFIG_MISSING'):
            fresh.release('alice', run['id'])
        with fresh.store.tx() as db:
            assert [tuple(r) for r in db.execute('SELECT * FROM reviews')] == old_rows
    finally:
        fresh.close()


def test_actual_sdk_fences_staged_review_on_configuration_change(tmp_path, snapshot, monkeypatch):
    binding, calls, stop = {}, [], Event()
    class Stalled(httpx.SyncByteStream):
        def __iter__(self):
            while not stop.wait(0.05):
                pass
            yield b''
    def provider(req):
        calls.append(True)
        if len(calls) == 3:
            monkeypatch.setitem(REVIEW_PROMPTS, 'medical', REVIEW_PROMPTS['medical'] + ' New rubric.')
            return httpx.Response(200, stream=Stalled())
        content = ({'type': 'tool_use', 'id': 'inspect', 'name': TOOL_PREFIX + 'inspect_review_packet', 'input': {}}
            if len(calls) == 1 else {'type': 'tool_use', 'id': 'submit', 'name': TOOL_PREFIX + 'submit_review',
                                   'input': dict(binding, verdict='pass', issues=[])})
        return httpx.Response(200, stream=SSE(message_events(content)))
    core, reviewer, run, hashes = prepare(tmp_path, snapshot, provider=provider)
    binding.update(hashes)
    try:
        with pytest.raises(GateError, match='REVIEW_CONFIG_CHANGED'):
            core.review('alice', run['id'], 'medical')
        assert count(core, 'reviews') == 0
        events = reviewer.evidence[0]['workers']
        assert events[-1]['exit_code'] == -9
        assert all(group_rss(e['pid']) == 0 for e in events if e['kind'] == 'worker_started')
    finally:
        stop.set()
        core.close()


@pytest.mark.parametrize('target', ['record_digest', 'record_schema', 'config_digest', 'config_body', 'revision'])
def test_corruption_blocks_already_published_read(bound, target):
    core, run = bound
    released = core.release('alice', run['id'])
    with core.store.tx() as db:
        if target == 'record_digest':
            db.execute("UPDATE reviews SET digest=NULL WHERE role='medical'")
        elif target == 'record_schema':
            row = db.execute("SELECT * FROM reviews WHERE role='medical'").fetchone()
            body = json.loads(row['body'])
            body['issues'] = ['not-a-permitted-issue']
            db.execute('UPDATE reviews SET body=?,digest=? WHERE id=?', (canonical(body), digest(body), row['id']))
        elif target == 'config_digest':
            db.execute("UPDATE review_configs SET digest='wrong' WHERE role='medical'")
        elif target == 'config_body':
            row = db.execute("SELECT * FROM review_configs WHERE role='medical'").fetchone()
            body = json.loads(row['body'])
            body['system_prompt'] += ' different criteria'
            db.execute("UPDATE review_configs SET body=?,digest=? WHERE role='medical'", (canonical(body), digest(body)))
        else:
            db.execute('UPDATE drafts SET revision=revision+1')
    expected = {'record_digest': 'REVIEW_INTEGRITY', 'record_schema': 'REVIEW_INVALID',
                'config_digest': 'REVIEW_CONFIG_INTEGRITY', 'config_body': 'REVIEW_CONFIG_CHANGED',
                'revision': 'DRAFT_REVISION_MISMATCH'}[target]
    with pytest.raises(GateError, match=expected):
        core.get_release('alice', released['release_id'])


def test_fixture_database_migration_preserves_legacy_review_columns(core, run, tmp_path):
    core.execute_models('alice', run['id'])
    job = core.create_draft_job('alice', run['id'])
    core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
    for role in ('medical', 'ethics'):
        core.review('alice', run['id'], role)
    core.close()
    with sqlite3.connect(core.store.path) as db:
        db.execute('DROP TABLE review_configs')
        db.execute('ALTER TABLE reviews DROP COLUMN digest')
        before = db.execute('SELECT * FROM reviews ORDER BY id').fetchall()
    fresh = Core(Store(core.store.path), enable_fixtures=True)
    try:
        with fresh.store.tx() as db:
            assert [tuple(row)[:-1] for row in db.execute('SELECT * FROM reviews ORDER BY id')] == before
            assert db.execute('SELECT count(*) FROM review_configs').fetchone()[0] == 0
        assert fresh.review('alice', run['id'], 'medical')['verdict'] == 'pass'
        released = fresh.release('alice', run['id'])
        assert fresh.get_release('alice', released['release_id'])['run_id'] == run['id']
    finally:
        fresh.close()


def test_sdk_rejects_wrong_configuration_before_any_provider_call():
    import asyncio
    from medical_harness.sdk_worker import run_session
    with pytest.raises(ValueError, match='configuration changed before worker startup'):
        asyncio.run(run_session({'port': 1, 'capability': 'offline', 'session': 'a'*32,
                                 'scenario': 'review_medical', 'review_config_hash': 'wrong'}))
