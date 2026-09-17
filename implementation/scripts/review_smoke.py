"""Two actual SDK/Flash reviewers on one synthetic engineering report; no publication."""

import argparse
import json
import os
from uuid import uuid4

from medical_harness.contracts import GateError, canonical
from medical_harness.core import Core
from medical_harness.paths import ROOT, confined
from medical_harness.review_agent import ReviewAgent
from medical_harness.store import Store
from medical_harness.workers import group_rss


def main(*, data=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.parse_args()
    os.umask(0o077)
    data = confined(data or ROOT / 'runtime/review-agent')
    reviewer = ReviewAgent(ROOT / 'runtime/private/deepseek.key', data)
    core = Core(Store(data / 'harness.sqlite3'), enable_fixtures=True, review_agent=reviewer)
    proof = {'clinical_validation': False, 'real_prediction_rl': False, 'desktop_agent_entry': False,
             'report_preparation': 'fixed sections plus trusted fixture renderer', 'results': []}
    try:
        case = core.create_case('review-smoke', json.loads((ROOT / 'examples/synthetic-case.json').read_text()))
        run = core.start_run('review-smoke', case['case_id'], 'eval', uuid4().hex)
        proof['run_id'] = run['id']
        core.execute_models('review-smoke', run['id'])
        job = core.create_draft_job('review-smoke', run['id'])
        draft = core.submit_proposal(job['job_id'], job['capability'], canonical({'sections': ['forecast', 'policy', 'limitations']}))
        proof['draft'] = draft
        for role in ('medical', 'ethics'):
            result = core.review('review-smoke', run['id'], role)
            proof['results'].append(result)
            if role == 'medical':
                try:
                    core.release('review-smoke', run['id'])
                except GateError as exc:
                    proof['single_review_release_denied'] = exc.code
                assert proof.get('single_review_release_denied') == 'REVIEW_REQUIRED'
        with core.store.tx() as db:
            reviews = [json.loads(row['body']) for row in db.execute('SELECT body FROM reviews WHERE run_id=?', (run['id'],))]
            assert len(reviews) == 2 and {r['role'] for r in reviews} == {'medical', 'ethics'}
            assert all(r['origin'] == 'model' and r['producer'].startswith('sdk.deepseek-flash-') for r in reviews)
            assert {r['report_hash'] for r in reviews} == {draft['report_hash']}
            assert {r['evidence_hash'] for r in reviews} == {job['evidence_hash']}
            proof['reviews'] = reviews
            proof['ledger'] = {table: db.execute('SELECT count(*) FROM ' + table + ' WHERE run_id=?', (run['id'],)).fetchone()[0]
                              for table in ('jobs', 'artifacts', 'drafts', 'reviews', 'releases')}
        assert proof['ledger'] == {'jobs': 5, 'artifacts': 2, 'drafts': 1, 'reviews': 2, 'releases': 0}
        sessions = [t['sdk']['results'][-1]['session_id'] for t in reviewer.evidence]
        assert len(set(sessions)) == 2
    finally:
        core.close()
        proof['review_agents'] = reviewer.evidence
        proof['workers_reaped'] = all(group_rss(e['pid']) == 0 for t in reviewer.evidence for e in t.get('workers', []) if e['kind'] == 'worker_started')
        proof['budget'] = json.loads((data / 'budget.json').read_text()) if (data / 'budget.json').exists() else None
        (data / 'result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    assert proof['workers_reaped']
    print(json.dumps(proof, ensure_ascii=False, indent=2))
    return proof


if __name__ == '__main__':
    main()
