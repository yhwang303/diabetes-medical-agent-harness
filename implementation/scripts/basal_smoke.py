"""Local HTTP plus independent workers; a fresh synthetic ledger, no paid provider."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4

import httpx

from medical_harness.paths import ROOT


def main():
    out = ROOT / 'runtime/basal-smoke' / uuid4().hex
    out.mkdir(parents=True, mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with (out / 'service.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-B', '-m', 'medical_harness.cli', 'serve',
            '--enable-fixtures', '--enable-basal-fixtures', '--port', str(port), '--data-dir', str(out)],
            cwd=ROOT, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=15) as client:
                for _ in range(50):
                    if process.poll() is not None:
                        raise RuntimeError('service exited before readiness')
                    try:
                        if client.get('/health').status_code == 200: break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError('service did not become ready')
                client.headers['Authorization'] = 'Bearer ' + (out / 'operator.token').read_text().strip()
                def request(method, path, body=None):
                    response = client.request(method, path, **({'json': body} if body is not None else {}))
                    response.raise_for_status()
                    return response.json()
                raw = json.loads((ROOT / 'examples/basal-research-case.json').read_text())
                case = request('POST', '/cases', raw)
                run = request('POST', '/runs', {'case_id': case['case_id'], 'mode': 'eval', 'idempotency_key': 'smoke'})
                base = '/runs/' + run['id']
                assert request('POST', base + '/execute')['state'] == 'SAFETY_ACCEPTED'
                assert client.post(base + '/release').json()['error'] == 'REVIEW_REQUIRED'
                job = request('POST', base + '/draft-jobs')
                response = client.post('/proposal-jobs/' + job['job_id'],
                    headers={'Authorization': 'Bearer ' + job['capability']},
                    json={'sections': ['forecast', 'policy', 'limitations']})
                response.raise_for_status()
                for role in ('medical', 'ethics'):
                    assert request('POST', base + '/reviews/' + role)['verdict'] == 'pass'
                released = request('POST', base + '/release')
                report = request('GET', '/releases/' + released['release_id'])
                slot = report['report']['sections'][1]
                conversion = slot['action_display']['conversion']
                assert (slot['data']['action']['action_value'], slot['data']['action']['action_unit']) == (0.6, 'U/h')
                assert (conversion['source_value'], conversion['source_unit']) == (0.6, 'U/h')
                assert conversion['exact']['rate_u_per_min'] == {'numerator': '1', 'denominator': '100'}
                assert slot['action_display']['interval_total_units'] == '0.05'
                assert request('GET', '/cases/' + case['case_id'] + '/timeline')['snapshot'] == raw
                events = request('GET', base + '/events')
                stops = [e['body'] for e in events if e['kind'] == 'worker_stopped']
                assert len(stops) == 4 and all(e['exit_code'] == 0 for e in stops)
                for worker in stops:
                    try: os.killpg(worker['pid'], 0)
                    except ProcessLookupError: pass
                    else: raise RuntimeError('worker group still alive')
                assert client.get('/releases/' + released['release_id'], headers={'Authorization': 'Bearer invalid'}).status_code == 401
                request('POST', base + '/cancel')
                assert client.get('/releases/' + released['release_id']).json()['error'] == 'CANCELLED'
                (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                (out / 'result.json').write_text(json.dumps({'run_id': run['id'], 'clinical_validation': False,
                    'model_calls': 'four independent fixture workers; no real numerical model or LLM',
                    'worker_groups_reaped': True, 'actual_history_unchanged': True,
                    'conversion_source_preserved': True, 'conversion': conversion,
                    'unauthorized_read': 401, 'post_cancel_read': 'CANCELLED', 'workers': stops}, indent=2) + '\n')
        finally:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    print(out / 'result.json')


if __name__ == '__main__':
    main()
