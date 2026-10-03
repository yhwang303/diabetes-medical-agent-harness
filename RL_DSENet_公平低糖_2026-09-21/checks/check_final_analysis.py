"""Independent raw trajectory arithmetic checks; never modifies scores/results."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
P = R.parent
sys.path.insert(0, str(R))
import final_analysis as fa


def independent(records, planned):
    rows = [r for r in records if not r['warmup']]
    out = {}
    for name in ['bg', 'cgm']:
        v = np.array([r[name + '_mg_dl'] for r in rows], dtype=float)
        g = v[np.isfinite(v) & (v > 0)]
        n = len(g)
        z = 1.509 * (np.log(np.clip(g, 20, 600)) ** 1.084 - 5.381)
        low = np.where(z <= 0, 10 * z ** 2, 0)
        high = np.where(z >= 0, 10 * z ** 2, 0)
        low = np.where(g <= 20, 100, np.where(g >= 600, 0, low))
        high = np.where(g >= 600, 100, np.where(g <= 20, 0, high))
        tir = int(((g >= 70) & (g <= 180)).sum())
        out[name] = dict(tir_lower_bound_pct=100*tir/planned,
                         tir_upper_bound_pct=100*(tir+planned-n)/planned,
                         coverage_pct=100*n/planned,
                         mean_mg_dl=float(g.mean()), sd_mg_dl=float(g.std(ddof=0)),
                         cv_pct=float(100*g.std(ddof=0)/g.mean()),
                         lbgi=float(low.mean()), hbgi=float(high.mean()))
        for key, cond in [('tbr70_pct', g < 70), ('tbr54_pct', g < 54),
                          ('tar180_pct', g > 180), ('tar250_pct', g > 250)]:
            out[name][key] = 100*float(cond.mean())
    days = len(rows)*5/1440
    rates = np.array([r['delivered_basal_u_h'] for r in rows])
    out.update(basal_u_per_observed_day=float(rates.sum()*5/60/days),
               bolus_u_per_observed_day=sum(r['bolus_u'] for r in rows)/days,
               action_tv_per_observed_day=float(np.abs(np.diff(rates)).sum())/days,
               pump_changed_fraction=float(np.mean([abs(r['requested_basal_u_h']-r['delivered_basal_u_h'])>1e-7 for r in rows])))
    return out


def check(folder):
    a = fa.read(folder)
    summary = json.loads((folder/'summary.json').read_text())
    patients = {}
    warmups = []
    for e in summary['episodes']:
        x = json.loads((folder/(e['key']+'.json')).read_text())
        records = x['records']
        assert [r['minute'] for r in records] == list(range(5, len(records)*5+1, 5))
        assert all(r['warmup'] == (r['minute'] <= 360) for r in records)
        warmups.append(sum(r['warmup'] for r in records))
        v = independent(records, (x['job']['total_minutes']-360)//5)
        for domain in ['bg', 'cgm']:
            for k in fa.G:
                np.testing.assert_allclose(v[domain][k], x['metrics'][domain][k], rtol=1e-12, atol=1e-12)
        for k in fa.D:
            np.testing.assert_allclose(v[k], x['metrics'][k], rtol=1e-12, atol=1e-12)
        patients.setdefault(str(x['job']['patient']), []).append(v)
    for k in fa.G + fa.D:
        pm = [np.mean([v['bg'][k] if k in fa.G else v[k] for v in cases]) for cases in patients.values()]
        np.testing.assert_allclose([a[k]['mean'], a[k]['sd']], [np.mean(pm), np.std(pm, ddof=1)], atol=1e-12)
    self_paired = fa.paired(a, a)
    assert all(v['ours_minus_other'] == 0 and v['patient_bootstrap_ci95'] == [0, 0] for v in self_paired.values())
    return {'folder':str(folder.relative_to(P)), 'episodes':len(summary['episodes']),
            'patient_clusters':len(patients), 'warmup_counts':sorted(set(warmups)),
            'raw_glucose_and_dose_arithmetic':'passed', 'patient_mean_sample_SD':'passed',
            'metrics':{k:a[k] for k in fa.G+fa.D}}


def main():
    folders = [Path(x).resolve() for x in sys.argv[1:]] or [P/'RL_DSENet_2026-09-17/results/F11_actor']
    results = [check(p) for p in folders]
    # Synthetic missing tails remain unknown; dose is per OBSERVED day.
    rows = [dict(warmup=False, bg_mg_dl=g, cgm_mg_dl=g,
                 delivered_basal_u_h=2., requested_basal_u_h=2., bolus_u=0.)
            for g in [60.,100.,200.]]
    m = fa.summarize(rows, 6, True)
    np.testing.assert_allclose([m['bg']['tir_lower_bound_pct'],m['bg']['tir_upper_bound_pct'],
                               m['bg']['coverage_pct'],m['bg']['tbr70_pct']], [100/6,400/6,50,100/3])
    np.testing.assert_allclose(m['basal_u_per_observed_day'],48.)
    retained = {}
    for role, item in json.loads((R/'configs/retained_weights.json').read_text()).items():
        p = P/'RL_DSENet_2026-09-17'/item['path']
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == item['sha256']
        retained[role] = dict(path=str(p.relative_to(P)), sha256=actual)
    result = dict(status='passed', folders=results, synthetic_missing_tail='passed',
                  retained_original_weights_verified=retained,
                  final_analysis_sha256=hashlib.sha256((R/'final_analysis.py').read_bytes()).hexdigest())
    (R/'checks/final_analysis_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
