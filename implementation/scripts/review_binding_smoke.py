"""Verify real reviewer bindings, then fault-inject private copies without more LLM calls."""

import json
import sqlite3
from unittest.mock import patch

from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core
from medical_harness.paths import ROOT
from medical_harness.review_agent import ReviewAgent
from medical_harness.sdk_worker import REVIEW_PROMPTS
from medical_harness.store import Store
from review_smoke import main as review_smoke


DATA = ROOT / 'runtime/review-binding'


def copy_core(name, source=None, *, clock=None):
    path = DATA / (name + '.sqlite3')
    if path.exists():
        raise RuntimeError('Evidence copy already exists; preserve it and use a fresh verification directory')
    with sqlite3.connect(source or DATA / 'harness.sqlite3') as src, sqlite3.connect(path) as dst:
        src.backup(dst)
    reviewer = ReviewAgent(ROOT / 'runtime/private/deepseek.key', DATA / 'unused')
    # No method below may start a new reviewer; never spend on fault-injection cases.
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected paid reviewer execution')
    reviewer.execute = forbidden
    return Core(Store(path), enable_fixtures=True, review_agent=reviewer, **({'clock': clock} if clock else {}))


def denied(call, expected):
    try:
        call()
    except GateError as exc:
        assert exc.code == expected, (exc.code, expected)
        return exc.code
    raise AssertionError('Expected gate refusal: ' + expected)


def legacy_rows(db):
    # All columns which existed before this migration; no new evidence invented.
    return {table: [list(row) for row in db.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('cases', 'snapshots', 'runs', 'jobs', 'artifacts', 'drafts', 'reviews', 'releases', 'events', 'outbox')}


def main():
    if (DATA / 'harness.sqlite3').exists():
        raise RuntimeError('Preserve existing live evidence; do not repeat paid verification implicitly')
    live = review_smoke(data=DATA)
    run_id, owner = live['run_id'], 'review-smoke'
    proof = {'run_id': run_id, 'clinical_validation': False, 'fault_injection_uses_paid_calls': False,
             'primary_live_releases': live['ledger']['releases'], 'bindings': []}
    budget_before = (DATA / 'budget.json').read_bytes()
    core = copy_core('restart')
    try:
        for agent in live['review_agents']:
            role = agent['role']
            assert core.review(owner, run_id, role)['verdict'] == 'pass'
            with core.store.tx() as db:
                row = db.execute('SELECT * FROM reviews WHERE run_id=? AND role=?', (run_id, role)).fetchone()
                config = db.execute('SELECT * FROM review_configs WHERE run_id=? AND role=?', (run_id, role)).fetchone()
                review = json.loads(row['body'])
                assert row['digest'] == digest(review)
                assert config['digest'] == digest(json.loads(config['body'])) == review['binding']['config_hash']
                assert agent['sdk']['review_config_hash'] == config['digest']
                proof['bindings'].append({'role': role, 'review_digest': row['digest'], **review['binding']})
        proof['restart_cached_reviews'] = 'both pass; zero new requests'
        released = core.release(owner, run_id)
        assert core.get_release(owner, released['release_id'])['report_hash'] == live['draft']['report_hash']
        proof['private_copy_release_and_read'] = 'passed'
        with patch.dict(REVIEW_PROMPTS, medical=REVIEW_PROMPTS['medical'] + ' changed rubric'):
            proof['prompt_change_read'] = denied(lambda: core.get_release(owner, released['release_id']), 'REVIEW_CONFIG_CHANGED')
    finally:
        core.close()
    core = copy_core('new-revision')
    try:
        job = core.create_draft_job(owner, run_id)
        current = core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
        assert current['report_hash'] == live['draft']['report_hash'] and current['draft_id'] != live['draft']['draft_id']
        proof['same_hash_new_revision'] = denied(lambda: core.release(owner, run_id), 'REVIEW_REQUIRED')
    finally:
        core.close()
    core = copy_core('forged-binding')
    try:
        with core.store.tx() as db:
            row = db.execute("SELECT * FROM reviews WHERE role='medical'").fetchone()
            body = json.loads(row['body'])
            body['binding']['role'] = 'ethics'
            db.execute('UPDATE reviews SET body=?,digest=? WHERE id=?', (canonical(body), digest(body), row['id']))
        proof['rehashed_wrong_role'] = denied(lambda: core.release(owner, run_id), 'REVIEW_INVALID')
    finally:
        core.close()
    legacy_path = ROOT / 'runtime/review-agent/harness.sqlite3'
    with sqlite3.connect(legacy_path) as db:
        before = legacy_rows(db)
        old_run = db.execute('SELECT id,owner,expires FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()
    core = copy_core('legacy-migration', legacy_path, clock=lambda: old_run[2] - 1)
    try:
        with core.store.tx() as db:
            after = legacy_rows(db)
            after['reviews'] = [row[:-1] for row in after['reviews']]
            assert after == before
            assert all(row['digest'] is None for row in db.execute('SELECT digest FROM reviews'))
        proof['legacy_real_missing_config'] = denied(lambda: core.release(old_run[1], old_run[0]), 'REVIEW_CONFIG_MISSING')
        proof['legacy_rows_preserved'] = {table: len(rows) for table, rows in before.items()}
        proof['legacy_check_clock'] = 'frozen before original expiry to isolate missing configuration'
    finally:
        core.close()
    assert (DATA / 'budget.json').read_bytes() == budget_before
    proof['budget_unchanged_by_offline_checks'] = True
    (DATA / 'binding-result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
