"""Explicit, budgeted real SDK/Flash smoke on a separate synthetic ledger."""

import argparse
import fcntl
import json
import os
from uuid import uuid4

from medical_harness.contracts import GateError
from medical_harness.core import Core
from medical_harness.flash_gateway import FlashGateway
from medical_harness.paths import ROOT, confined
from medical_harness.sdk_worker import SDK_EXECUTOR
from medical_harness.store import Store
from medical_harness.workers import WorkerLimits, WorkerRunner, group_rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--scenario', choices=['tools', 'subagent', 'interrupt'], required=True)
    args = parser.parse_args()
    os.umask(0o077)
    data = confined(ROOT / 'runtime/sdk-smoke')
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (data / 'smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key_path = confined(ROOT / 'runtime/private/deepseek.key')
        if key_path.stat().st_mode & 0o077:
            raise SystemExit('Key file must be private (0600).')
        core = Core(Store(data / 'harness.sqlite3'), enable_fixtures=True)
        case = core.create_case('sdk-smoke', json.loads((ROOT / 'examples/synthetic-case.json').read_text()))
        run = core.start_run('sdk-smoke', case['case_id'], 'eval', uuid4().hex)
        tools_seen = []

        def invoke_tool(name):
            try:
                if name == 'prediction_probe':
                    core.execute_prediction('sdk-smoke', run['id'])
                    result = {'ok': True, 'origin': 'fixture', 'clinical_use': False}
                elif name == 'release_probe':
                    core.release('sdk-smoke', run['id'])
                    raise AssertionError('Release gate unexpectedly opened')
                else:
                    evidence = core.case_evidence('sdk-smoke', case['case_id'])
                    result = {'ok': True, 'clinical_use': False,
                              'cards': [{'kind': c['kind'], 'status': c['status']} for c in evidence['cards']]}
            except GateError as exc:
                core.record_denial('sdk-smoke', run['id'], exc.code)
                result = {'ok': False, 'error': exc.code, 'clinical_use': False}
            tools_seen.append({'name': name, **result})
            return result

        events = []
        runner = WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
                              file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)
        proof = {'scenario': args.scenario, 'run_id': run['id'], 'clinical_validation': False,
                 'real_prediction_rl': False, 'desktop_agent_entry': False, 'tools': tools_seen, 'worker_events': events}
        try:
            with FlashGateway(key_path.read_text().strip(), data / 'budget.json', invoke_tool) as gateway:
                proof['provider'] = gateway.records
                try:
                    proof['sdk'] = runner.run(SDK_EXECUTOR,
                        {'port': gateway.port, 'capability': gateway.capability, 'scenario': args.scenario, 'session': uuid4().hex},
                        timeout=90, validate=lambda: None, job_id=uuid4().hex,
                        notify=lambda kind, **details: events.append({'kind': kind, **details}))
                except GateError as exc:
                    proof['error'] = exc.code
        finally:
            runner.close()
            core.close()
            proof['budget'] = json.loads((data / 'budget.json').read_text()) if (data / 'budget.json').exists() else None
            proof['workers_reaped'] = all(group_rss(e['pid']) == 0 for e in events if e['kind'] == 'worker_started')
            with core.store.tx() as db:
                proof['ledger'] = {table: db.execute('SELECT count(*) FROM ' + table + ' WHERE run_id=?', (run['id'],)).fetchone()[0]
                                   for table in ('jobs', 'artifacts', 'drafts', 'reviews', 'releases')}
            (data / (args.scenario + '-result.json')).write_text(json.dumps(proof, indent=2, ensure_ascii=False))
        print(json.dumps(proof, ensure_ascii=False, indent=2))
        assert 'error' not in proof, proof.get('error')
        assert proof['workers_reaped'] and proof['sdk']['stream_events'] > 0
        assert proof['ledger']['releases'] == 0 and proof['ledger']['drafts'] == 0
        assert any(record.get('model_responses') for record in proof['provider'])
        assert not any(record.get('error') for record in proof['provider'])
        if args.scenario != 'interrupt':
            assert proof['sdk']['results'][-1]['terminal_reason'] == 'completed'
            assert not proof['sdk']['results'][-1]['is_error']
        if args.scenario == 'tools':
            assert [(x['name'], x['ok']) for x in tools_seen] == [('prediction_probe', True), ('release_probe', False)]
            assert proof['ledger']['artifacts'] == 1
        elif args.scenario == 'subagent':
            assert tools_seen and all(x['name'] == 'inspect_evidence' for x in tools_seen)
            assert any(x['event'] == 'SubagentStart' for x in proof['sdk']['hooks'])
            assert any(x['event'] == 'SubagentStop' for x in proof['sdk']['hooks'])
        else:
            assert proof['sdk']['interrupted']
            assert proof['sdk']['results'][-1]['terminal_reason'] in ('aborted_streaming', 'aborted_tools')


if __name__ == '__main__':
    main()
