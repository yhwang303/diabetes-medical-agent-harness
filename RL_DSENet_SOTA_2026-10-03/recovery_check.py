"""Recover the frozen actor on the new machine without writing old outputs."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent
OLD = P / 'RL_DSENet_公平低糖_2026-09-21'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = R / 'checks'
    checks.mkdir(exist_ok=True)
    weights = json.loads((OLD/'configs/retained_weights.json').read_text())
    for item in weights.values():
        assert sha(P/'RL_DSENet_2026-09-17'/item['path']) == item['sha256']
    cfg = json.loads((OLD/'configs/confirmation.json').read_text())
    cfg['patients'] = [1, 9]
    cfg['groups'] = [dict(name='over_bolus', bolus_factor=1.2,
                          seeds=[92111], irregular=False)]
    cfg['comparison_status'] = 'recovery only, exposed old cases; not new confirmation'
    path = checks/'recovery_config.json'
    path.write_text(json.dumps(cfg, indent=2))
    # The old evaluator and scorer are imported verbatim. Only output ROOT changes.
    sys.path.insert(0, str(OLD))
    import evaluate_final as evaluator
    evaluator.ROOT = R
    (R/'results').mkdir(exist_ok=True)
    sys.argv = [__file__, '--config', str(path), '--checkpoint',
                str(P/'RL_DSENet_2026-09-17/results/D06_selected_policy/policy.pt'),
                '--mode', 'actor', '--bounded-reference', '--name', 'recovery_actor',
                '--batch-size', '2']
    evaluator.main()
    compared = []
    for patient in cfg['patients']:
        name = 'over_bolus_p%02d_s92111.json' % patient
        old = json.loads((OLD/'results/C_native_ours'/name).read_text())
        new = json.loads((R/'results/recovery_actor'/name).read_text())
        assert old['job'] == new['job']
        assert old['records'] == new['records'], name
        assert old['metrics'] == new['metrics'], name
        assert old['failure_reason'] == new['failure_reason'] is None
        compared.append(dict(case=name, records_exact=True, metrics_exact=True,
                             old_sha256=sha(OLD/'results/C_native_ours'/name),
                             new_sha256=sha(R/'results/recovery_actor'/name)))
    result = dict(status='passed', frozen_weights=weights, compared=compared,
                  python=sys.version, evaluator_sha256=sha(OLD/'evaluate_final.py'),
                  scorer_sha256=sha(P/'RL_DITR创新_2026-09-16/control_metrics.py'),
                  numpy=__import__('numpy').__version__,
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                               '--format=csv,noheader'], text=True).strip())
    (checks/'recovery.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
