"""Real SDK report proposal -> trusted renderer on synthetic fixture evidence only."""

import argparse
import json
import os
from uuid import uuid4

from medical_harness.contracts import GateError, digest
from medical_harness.core import Core
from medical_harness.paths import ROOT, confined
from medical_harness.report_agent import ReportAgent
from medical_harness.store import Store
from medical_harness.workers import group_rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.parse_args()
    os.umask(0o077)
    data = confined(ROOT / 'runtime/report-agent')
    core = Core(Store(data / 'harness.sqlite3'), enable_fixtures=True)
    agent = ReportAgent(core, ROOT / 'runtime/private/deepseek.key', data)
    proof = {'real_prediction_rl': False, 'clinical_validation': False, 'desktop_agent_entry': False}
    try:
        case = core.create_case('report-smoke', json.loads((ROOT / 'examples/synthetic-case.json').read_text()))
        run = core.start_run('report-smoke', case['case_id'], 'eval', uuid4().hex)
        assert core.execute_models('report-smoke', run['id'])['state'] == 'SAFETY_ACCEPTED'
        result = agent.generate('report-smoke', run['id'])
        proof['result'] = result
        assert result['state'] == 'DRAFT_READY'
        assert set(result) == {'draft_id', 'report_hash', 'state'}
        with core.store.tx() as db:
            row = db.execute('SELECT * FROM drafts WHERE id=?', (result['draft_id'],)).fetchone()
            body = json.loads(row['body'])
            assert digest(body) == row['digest'] == result['report_hash']
            assert body['origin'] == 'fixture' and body['clinical_use'] is False
            assert {s['kind'] for s in body['sections']} == {'forecast', 'policy', 'limitations'}
            for section in body['sections']:
                if section['kind'] in ('forecast', 'policy'):
                    artifact = db.execute('SELECT body FROM artifacts WHERE id=? AND run_id=?',
                        (section['artifact_id'], run['id'])).fetchone()
                    assert section['data'] == json.loads(artifact['body'])['payload']
            proof['ledger'] = {table: db.execute('SELECT count(*) FROM ' + table + ' WHERE run_id=?', (run['id'],)).fetchone()[0]
                              for table in ('jobs', 'artifacts', 'drafts', 'reviews', 'releases')}
            proof['report_verification'] = {'hash_matches': True, 'artifact_values_exact': True,
                'origin': body['origin'], 'clinical_use': body['clinical_use'], 'template_version': body['template_version']}
        try:
            core.release('report-smoke', run['id'])
        except GateError as exc:
            proof['release_denied'] = exc.code
        assert proof.get('release_denied') == 'REVIEW_REQUIRED'
        assert proof['ledger'] == {'jobs': 3, 'artifacts': 2, 'drafts': 1, 'reviews': 0, 'releases': 0}
    finally:
        agent.runner.close()
        core.close()
        proof['agent'] = agent.evidence
        proof['workers_reaped'] = all(group_rss(e['pid']) == 0 for e in agent.evidence.get('workers', []) if e['kind'] == 'worker_started')
        proof['budget'] = json.loads((data / 'budget.json').read_text()) if (data / 'budget.json').exists() else None
        (data / 'result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2))
    assert proof['workers_reaped']
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
