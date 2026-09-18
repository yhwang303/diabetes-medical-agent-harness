"""Verified, patient-equal comparison of frozen external methods (no training).

Run in the same remote .venv Python/NumPy runtime as control_metrics.summarize.
Missing outcomes remain missing; uncertainty excludes training-seed variation.
"""
import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'RL_DITR创新_2026-09-16'))
from control_metrics import summarize

GLUCOSE = ('tir_lower_bound_pct', 'tir_upper_bound_pct', 'tir_observed_pct',
           'tbr70_pct', 'tbr54_pct', 'tar180_pct', 'tar250_pct', 'mean_mg_dl',
           'sd_mg_dl', 'cv_pct', 'lbgi', 'hbgi', 'risk', 'coverage_pct',
           'hypo_events_per_observed_day')
ACTION = ('basal_u_per_observed_day', 'bolus_u_per_observed_day',
          'action_tv_per_observed_day', 'pump_changed_fraction')
DISPLAY = [('tbr70_pct', 'TBR70'), ('tbr54_pct', 'TBR54'),
           ('tar180_pct', 'TAR180'), ('tar250_pct', 'TAR250'),
           ('mean_mg_dl', 'Mean BG'), ('cv_pct', 'CV'), ('lbgi', 'LBGI'), ('hbgi', 'HBGI')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(values):
    good = [float(v) for v in values if v is not None]
    return float(np.mean(good)) if good else None


def stats(values):
    good = [float(v) for v in values if v is not None]
    return {'mean': average(good),
            'sd_between_patients': float(np.std(good, ddof=1)) if len(good) > 1 else None,
            'n_patients': len(good)}


def read_method(method, folder_key):
    folder = PROJECT / method[folder_key]
    summary = json.loads((folder / 'summary.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert summary['status'] == 'completed' and len(summary['episodes']) == 60, folder
    jobs = manifest['jobs']
    assert len(jobs) == 60
    expected = {(j['patient'], j['group'], j['seed']): j for j in jobs}
    assert len(expected) == 60
    checkpoint = method.get('checkpoint_sha256', method.get('sha256'))
    if method.get('checkpoint'):
        assert sha(PROJECT / method['checkpoint']) == manifest['checkpoint_sha256'], (method['key'], 'current frozen weights mismatch')
    if checkpoint is not None:
        assert checkpoint == manifest['checkpoint_sha256'], (method['key'], 'checkpoint hash')
    patients = sorted({j['patient'] for j in jobs})
    assert len(patients) == 10
    collected = {p: [] for p in patients}
    raws = {}
    raw_hashes = {}
    failures = []
    for episode in summary['episodes']:
        path = folder / (episode['key'] + '.json')
        raw = json.loads(path.read_text())
        ident = (episode['patient'], episode['group'], episode['seed'])
        assert ident in expected and ident not in raws
        assert raw['job'] == expected[ident], (method['key'], ident, 'job mismatch')
        planned = (raw['job']['total_minutes'] - 360) // 5
        metrics = summarize(raw['records'], planned, raw['failure_reason'] is not None)
        assert metrics == raw['metrics'] == episode['metrics'], (
            method['key'], episode['key'], 'exact raw metric recomputation mismatch; use simulator runtime')
        assert episode['failure_reason'] == raw['failure_reason']
        if folder_key == 'main_folder':
            warmup = [r for r in raw['records'] if r['warmup']]
            if any(not r['warmup'] for r in raw['records']):
                assert warmup, 'Missing common warmup'
                anchor = warmup[-1]['delivered_basal_u_h']
                assert all(abs(r['requested_basal_u_h'] - anchor) <= .25001
                           for r in raw['records'] if not r['warmup']), (method['key'], 'outside common action envelope')
        raws[ident] = raw
        raw_hashes[path.name] = sha(path)
        collected[episode['patient']].append(metrics)
        if raw['failure_reason'] is not None:
            failures.append({'key': episode['key'], 'reason': raw['failure_reason']})
    assert set(raws) == set(expected)

    projections = {ident: [] for ident in expected}
    projection_path = folder / 'action_projection.json'
    if projection_path.exists():
        for r in json.loads(projection_path.read_text()):
            ident = (r['patient'], r['group'], r['seed'])
            assert ident in projections
            projections[ident].append(abs(r['raw_u_h'] - r['projected_u_h']) > 1e-7)
    if manifest.get('common_limit_u_h') is not None:
        assert projection_path.exists(), 'Projected evaluation lacks action audit'
        for ident, raw in raws.items():
            # An aborted action may produce an extra decision, but every applied
            # control action must be present in the external projection audit.
            count = sum(not r['warmup'] for r in raw['records'])
            assert len(projections[ident]) >= count, (ident, 'missing projection decisions')

    patient_rows = {}
    for patient, rows in collected.items():
        assert len(rows) == 6, (patient, len(rows))
        record = {source: {k: average([r[source].get(k) for r in rows]) for k in GLUCOSE}
                  for source in ('bg', 'cgm')}
        record.update({k: average([r.get(k) for r in rows]) for k in ACTION})
        scenario_projection = []
        for ident in expected:
            if ident[0] == patient:
                vals = projections[ident]
                scenario_projection.append(100 * float(np.mean(vals)) if vals else (
                    None if manifest.get('common_limit_u_h') is not None else 0.))
        record['projection_pct'] = average(scenario_projection)
        record['failures'] = sum(r['failed'] for r in rows)
        record['prolonged_under54_120min_events'] = sum(
            r['bg'].get('prolonged_under54_120min_events', 0) for r in rows)
        record['hypo_events'] = sum(len(r['bg'].get('hypo_events', [])) for r in rows)
        record['right_censored_hypo_events'] = sum(
            r['bg'].get('right_censored_hypo_events', 0) for r in rows)
        record['max_hypo_low_minutes'] = max(r['bg'].get('max_hypo_low_minutes', 0) for r in rows)
        patient_rows[str(patient)] = record
    aggregate = {source: {k: stats([r[source][k] for r in patient_rows.values()]) for k in GLUCOSE}
                 for source in ('bg', 'cgm')}
    aggregate.update({k: stats([r[k] for r in patient_rows.values()]) for k in ACTION + ('projection_pct',)})
    aggregate.update(failures=len(failures), episodes=60,
                     prolonged_under54_120min_events=sum(r['prolonged_under54_120min_events'] for r in patient_rows.values()),
                     hypo_events=sum(r['hypo_events'] for r in patient_rows.values()),
                     right_censored_hypo_events=sum(r['right_censored_hypo_events'] for r in patient_rows.values()),
                     max_hypo_low_minutes=max(r['max_hypo_low_minutes'] for r in patient_rows.values()))
    for k in ('tbr70_pct', 'tbr54_pct'):
        valid = [(p, r['bg'][k]) for p, r in patient_rows.items() if r['bg'][k] is not None]
        aggregate['worst_patient_' + k] = max(valid, key=lambda pair: pair[1]) if valid else None
    return {'metadata': method, 'folder': method[folder_key], 'aggregate': aggregate,
            'patients': patient_rows, 'failure_reasons': failures,
            'manifest_sha256': sha(folder / 'manifest.json'), 'summary_sha256': sha(folder / 'summary.json'),
            'raw_sha256': raw_hashes, 'checkpoint_sha256': manifest.get('checkpoint_sha256'),
            'common_limit_u_h': manifest.get('common_limit_u_h'),
            'wall_seconds': summary.get('wall_seconds')}, expected


def bootstrap(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return None
    rng = np.random.default_rng(260917)
    draws = a[rng.integers(len(a), size=(10000, len(a)))].mean(1)
    return {'difference': float(a.mean()), 'ci95': np.percentile(draws, [2.5, 97.5]).tolist(),
            'patients': len(a)}


def exact_sign_flip(a):
    a = np.asarray(a, dtype=float)
    assert len(a) == 10
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(a))))
    observed = abs(float(a.mean()))
    return float(np.mean(np.abs((signs * a).mean(1)) >= observed - 1e-12))


def paired_analysis(methods, primary_key, hold_key):
    primary = methods[primary_key]
    comparisons = {}
    for key, other in methods.items():
        if key == primary_key:
            continue
        pairs = [(primary['patients'][p]['bg'], other['patients'][p]['bg'])
                 for p in primary['patients']]
        result = {metric: bootstrap([a[metric] - b[metric] for a, b in pairs
                                    if a[metric] is not None and b[metric] is not None])
                  for metric in ('tbr70_pct', 'tbr54_pct', 'tar180_pct', 'risk')}
        result['tir_difference_lower_bound'] = bootstrap([
            a['tir_lower_bound_pct'] - b['tir_upper_bound_pct'] for a, b in pairs])
        result['tir_difference_upper_bound'] = bootstrap([
            a['tir_upper_bound_pct'] - b['tir_lower_bound_pct'] for a, b in pairs])
        complete = all(abs(a['tir_upper_bound_pct'] - a['tir_lower_bound_pct']) < 1e-10
                       and abs(b['tir_upper_bound_pct'] - b['tir_lower_bound_pct']) < 1e-10
                       for a, b in pairs)
        result['tir_exact_sign_flip_p'] = exact_sign_flip([
            a['tir_lower_bound_pct'] - b['tir_lower_bound_pct'] for a, b in pairs]) if complete else None
        result['test_status'] = 'complete paired TIR' if complete else 'not testable: missing TIR outcomes'
        result['in_holm_family'] = key != hold_key
        comparisons[key] = result
    family = [(key, r['tir_exact_sign_flip_p'] if r['tir_exact_sign_flip_p'] is not None else 1.)
              for key, r in comparisons.items() if r['in_holm_family']]
    running = 0.
    for rank, (key, p) in enumerate(sorted(family, key=lambda v: (v[1], v[0]))):
        running = max(running, min(1., (len(family) - rank) * p))
        r = comparisons[key]
        r['tir_holm_adjusted_p'] = running if r['tir_exact_sign_flip_p'] is not None else None
        r['holm_family_size'] = len(family)
    return comparisons


def latex_escape(text):
    return str(text).replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')


def formatted(stat, tex=False):
    if stat['mean'] is None:
        return '--'
    if stat['sd_between_patients'] is None:
        return '%.2f' % stat['mean']
    return ('%.2f $\\pm$ %.2f' if tex else '%.2f ± %.2f') % (stat['mean'], stat['sd_between_patients'])


def table_rows(methods, tex=False):
    rows = []
    for result in methods.values():
        meta, a = result['metadata'], result['aggregate']
        g = a['bg']; lo, hi = g['tir_lower_bound_pct']['mean'], g['tir_upper_bound_pct']['mean']
        incomplete = hi - lo > 1e-10
        tir = ('[%.2f, %.2f]' % (lo, hi)) if incomplete or a['failures'] else formatted(g['tir_lower_bound_pct'], tex)
        rows.append([latex_escape(meta['label']) if tex else meta['label'],
                     latex_escape('%s / %s' % (meta['year'], meta['venue'])) if tex else '%s / %s' % (meta['year'], meta['venue']),
                     'L+S' if meta['key'] == 'ours' else ('warmup' if meta['key'] == 'hold' else 'L'),
                     tir] + [formatted(g[k], tex) for k, _ in DISPLAY] + [
                         '%d/60' % a['failures'], formatted(g['coverage_pct'], tex), formatted(a['projection_pct'], tex)])
    return rows


def write_tables(out, name, methods):
    headers = ['Method', 'Year / venue', 'Training data', 'TIR70-180 % (SD or missing bounds)'] + [
        label + (' mg/dL' if k == 'mean_mg_dl' else (' %' if k.endswith('_pct') else '')) for k, label in DISPLAY]
    headers += ['Failures / episodes', 'Coverage %', 'Projection %']
    with (out / (name + '_comparison.csv')).open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(table_rows(methods))
    tex = ['% Requires booktabs and graphicx; values are mean +/- between-patient SD (n=10).',
           '\\begin{table*}[t]', '\\centering', '\\scriptsize', '\\resizebox{\\textwidth}{!}{%',
           '\\begin{tabular}{ll' + 'r' * (len(headers) - 2) + '}', '\\toprule',
           ' & '.join(latex_escape(x) for x in headers) + ' \\\\', '\\midrule']
    tex += [' & '.join(row) + ' \\\\' for row in table_rows(methods, True)]
    tex += ['\\bottomrule', '\\end{tabular}}',
            '\\caption{Closed-loop control on 10 previously exposed virtual adults, with six matched scenarios each. '
            'One training seed. Values are patient-equal means and between-patient standard deviations. '
            'Bracketed TIR values are missing-outcome bounds, not confidence intervals. '
            'Other glucose outcomes are observed-only; coverage and failures are reported explicitly. '
            'Projection denotes external action clipping, not pump quantization. '
            'L: Loop only. L+S: Loop plus inherited simulation-trained context and paired simulator supervision. '
            'This compares control systems under the same evaluation protocol, not algorithms with identical training information. '
            '*RL-DITR is a continuous-basal task adaptation, not a reproduction of the original clinical trial. '
            + ('All methods share an observed-warmup basal envelope of plus/minus 0.25 U/h. ' if name == 'main' else 'Native action supports differ. ')
            + 'Methods are not guaranteed equally converged under the fixed budgets.}',
            '\\label{tab:' + name + '-control}', '\\end{table*}']
    (out / (name + '_comparison.tex')).write_text('\n'.join(tex) + '\n')
    # A compact double-column version avoids long CSV headings shrinking the
    # scientific values to unreadable type. Full details remain in the wide file.
    compact_headers = ['Method (year)', 'Data', 'TIR $\\uparrow$',
                       'TBR$_{70}$ $\\downarrow$', 'TBR$_{54}$ $\\downarrow$',
                       'TAR$_{180}$ $\\downarrow$', 'TAR$_{250}$ $\\downarrow$',
                       'CV $\\downarrow$', 'LBGI $\\downarrow$', 'HBGI $\\downarrow$', 'Fail.']
    compact = ['% Requires booktabs and graphicx. Detailed coverage/projection: full comparison table.',
               '\\begin{table*}[t]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
               '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{lcrrrrrrrrr}', '\\toprule',
               ' & '.join(compact_headers) + ' \\\\', '\\midrule']
    for raw, result in zip(table_rows(methods, True), methods.values()):
        year = result['metadata']['year']
        label = raw[0] + ((' (' + str(year) + ')') if isinstance(year, int) else '')
        # Full row: method, venue, data, TIR, TBR70, TBR54, TAR180, TAR250,
        # mean BG, CV, LBGI, HBGI, failures, coverage, external projection.
        selected = [label] + [raw[i] for i in (2,3,4,5,6,7,9,10,11,12)]
        compact.append(' & '.join(selected) + ' \\\\')
    compact += tex[-5:]
    compact[-3] = compact[-3].replace('Methods are not guaranteed', 'Full coverage, mean glucose and external projection rates are provided in the full supplementary table. Methods are not guaranteed')
    compact[-2] = '\\label{tab:' + name + '-control-compact}'
    (out / (name + '_comparison_compact.tex')).write_text('\n'.join(compact) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'configs/comparison_manifest.json'))
    args = parser.parse_args(); path = Path(args.manifest)
    config = json.loads(path.read_text())
    methods = config if isinstance(config, list) else config['methods']
    old_freeze = json.loads((ROOT.parent/'configs/final_freeze.json').read_text())
    scorer_relative = 'RL_DITR创新_2026-09-16/control_metrics.py'
    assert sha(PROJECT/scorer_relative) == old_freeze['source_sha256'][scorer_relative]
    assert len({m['key'] for m in methods}) == len(methods)
    assert all(m.get('category') != 'internal' for m in methods), 'Internal variants do not belong in the external comparison'
    primary = 'ours' if isinstance(config, list) else config.get('primary_key', config.get('primary_method', 'ours'))
    hold = 'hold' if isinstance(config, list) else config.get('hold_key', 'hold')
    out = ROOT / 'paper_tables'; out.mkdir(exist_ok=True)
    result = {'manifest_sha256': sha(path), 'numpy_version': np.__version__,
              'scorer_sha256': sha(PROJECT / 'RL_DITR创新_2026-09-16/control_metrics.py'),
              'single_training_seed': 260915, 'patients': 10, 'scenarios_per_patient': 6,
              'bootstrap_replicates': 10000, 'bootstrap_seed': 260917,
              'statistical_unit': 'patient mean of six paired scenarios; SD is between-patient sample SD (ddof=1)',
              'statistical_limits': 'CI excludes training-seed variation. TBR/risk CI is descriptive and observed-only. '
              'TIR sign-flip tests require complete outcomes; unavailable comparisons retain a p=1 slot in the fixed Holm family. '
              'Two-sided exact 1024 sign flips assume exchangeable/symmetric paired patient differences; '
              'the virtual cohort was previously exposed, not unseen-patient generalization.',
              'panels': {}}
    paired_csv = []; risk_rows = []; common_jobs = None
    for panel in ('main', 'native'):
        entries = {}
        for method in methods:
            entry, jobs = read_method(method, panel + '_folder')
            if common_jobs is None:
                common_jobs = jobs
            assert jobs == common_jobs, (method['key'], panel, 'unmatched scenarios')
            if panel == 'native':
                assert entry['checkpoint_sha256'] == result['panels']['main']['methods'][method['key']]['checkpoint_sha256'], (
                    method['key'], 'main/native checkpoint mismatch')
            entries[method['key']] = entry
        assert primary in entries and hold in entries
        comparisons = paired_analysis(entries, primary, hold)
        result['panels'][panel] = {'methods': entries, 'paired_comparisons': comparisons}
        write_tables(out, panel, entries)
        for key, comp in comparisons.items():
            row = {'panel': panel, 'primary': primary, 'comparator': key, 'test_status': comp['test_status'],
                   'tir_p': comp['tir_exact_sign_flip_p'], 'tir_p_holm': comp.get('tir_holm_adjusted_p'),
                   'holm_family_size': comp.get('holm_family_size'), 'in_holm_family': comp['in_holm_family']}
            for metric in ('tir_difference_lower_bound', 'tir_difference_upper_bound', 'tbr70_pct', 'tbr54_pct', 'tar180_pct', 'risk'):
                data = comp[metric]
                row[metric + '_delta'] = data['difference'] if data else None
                row[metric + '_ci_low'] = data['ci95'][0] if data else None
                row[metric + '_ci_high'] = data['ci95'][1] if data else None
            paired_csv.append(row)
        for key, entry in entries.items():
            a = entry['aggregate']; g = a['bg']
            row = {'panel': panel, 'method': key, 'hypo_events_total': a['hypo_events'],
                   'hypo_events_per_day': g['hypo_events_per_observed_day']['mean'],
                   'prolonged_under54_120min_events': a['prolonged_under54_120min_events'],
                   'right_censored_hypo_events': a['right_censored_hypo_events'],
                   'max_hypo_low_minutes': a['max_hypo_low_minutes']}
            for k in ('tbr70_pct', 'tbr54_pct'):
                worst = a['worst_patient_' + k]
                row['worst_patient_' + k] = worst[0] if worst else None
                row['worst_value_' + k] = worst[1] if worst else None
            row.update({k: a[k]['mean'] for k in ACTION})
            risk_rows.append(row)
    result['all_raw_metrics_recomputed_exactly'] = True
    result['all_60_jobs_matched_across_methods_and_panels'] = True
    (out / 'extended_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for name, rows in [('paired_comparisons.csv', paired_csv), ('risk_extra.csv', risk_rows)]:
        with (out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'status': 'verified_analysis_complete', 'methods': len(methods),
                      'panels': 2, 'table_episode_entries': 120 * len(methods), 'unique_result_folders': len({m[k] for m in methods for k in ('main_folder','native_folder')}), 'unique_episode_records': 60 * len({m[k] for m in methods for k in ('main_folder','native_folder')}), 'output': str(out)}))


if __name__ == '__main__':
    main()
