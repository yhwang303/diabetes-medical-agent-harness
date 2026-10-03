"""Run every already-frozen confirmation method once, at most two in parallel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import confirmation_gate as gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    freeze = args.freeze.resolve()
    frozen = gate.load_freeze(freeze)
    if gate.events(freeze.parent):
        raise ValueError('A fresh frozen set is required; no queue resume or automatic retries')
    audit = gate.R / 'checks' / ('confirmation_queue_' + freeze.parent.name)
    audit.mkdir(exist_ok=False)
    methods = [item['id'] for item in frozen['plan']['methods']]
    gate.write_new(audit / 'plan.json', dict(freeze_path=str(freeze), freeze_sha256=gate.sha(freeze),
        script_sha256=gate.sha(Path(__file__)), method_order=methods, max_parallel=2,
        batch_sizes={m['id']:m['batch_size'] for m in frozen['plan']['methods']},
        automatic_retry=False, scores_used_to_change_schedule=False,
        started_utc=datetime.now(timezone.utc).isoformat()))
    # Authorize the entire fixed set before any child can produce outcomes.
    tickets = [(mid, gate.authorize(freeze, mid)) for mid in methods]
    runner = gate.file_binding(frozen['plan']['confirmation_runner'])

    def execute(item):
        mid, ticket = item
        log = audit / (mid + '.log')
        command = [sys.executable, '-u', str(runner), '--ticket', str(ticket)]
        print(json.dumps(dict(event='started', method=mid, command=command)), flush=True)
        launch_error = None
        try:
            with log.open('x') as stream:
                code = subprocess.run(command, cwd=gate.P, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
        except Exception as error:
            code = None
            launch_error = repr(error)
        finished = [e for e in gate.events(freeze.parent)
                    if e.get('method_id') == mid and e['event'] == 'finished']
        if not finished:
            # Preserve an infrastructure failure; never rerun the consumed ticket.
            finished = [gate.record_result(ticket, failure_reason=
                'Confirmation process ended without a finished record: exit=%r error=%s' % (code, launch_error))]
        if len(finished) != 1:
            raise ValueError('Expected exactly one finished event: ' + mid)
        result = dict(method_id=mid, exit_code=code, launch_error=launch_error,
                      status=finished[0]['status'], error=finished[0]['error'],
                      complete_fixed_case_reporting=finished[0]['complete_fixed_case_reporting'],
                      ticket_sha256=gate.sha(ticket), log_sha256=gate.sha(log) if log.exists() else None)
        gate.write_new(audit / (mid + '.json'), result)
        print(json.dumps(dict(event='finished', **result)), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, tickets))
    gate.write_new(audit / 'summary.json', dict(status='fixed_set_executed', methods=results,
        all_methods_reported=set(methods) == {r['method_id'] for r in results},
        finished_utc=datetime.now(timezone.utc).isoformat(), no_model_promotion=True))
    print(json.dumps(dict(event='queue_finished', methods=len(results), audit=str(audit))), flush=True)
    if any(r['exit_code'] != 0 or r['error'] is not None for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
