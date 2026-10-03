"""Read-only score verification and patient-level tables for matched result panels.

Example: python summarize_panels.py --panel D_hold --panel D_legacy \
    --reference D_hold --output checks/panels_development.json
The first panel is the reference unless --reference is given. No ranking or SOTA
claim is produced. Failed/incomplete panels have Observed tables only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
SCORER = P/'RL_DITR创新_2026-09-16/control_metrics.py'
sys.path.insert(0, str(SCORER.parent))
from control_metrics import summarize

GLUCOSE = [
    'tir_observed_pct', 'tir_lower_bound_pct', 'tir_upper_bound_pct',
    'tbr70_pct', 'tbr54_pct', 'tar180_pct', 'tar250_pct', 'mean_mg_dl',
    'sd_mg_dl', 'cv_pct', 'lbgi', 'hbgi', 'risk', 'coverage_pct',
    'prolonged_under54_120min_events', 'hypo_events_per_observed_day',
    'max_hypo_low_minutes', 'right_censored_hypo_events',
]
DOSE = ['basal_u_per_observed_day', 'bolus_u_per_observed_day',
        'action_tv_per_observed_day', 'pump_changed_fraction']
OLD_METRICS = ['tir_lower_bound_pct', 'tir_upper_bound_pct', 'tbr70_pct',
               'tbr54_pct', 'tar180_pct', 'tar250_pct', 'sd_mg_dl', 'cv_pct',
               'lbgi', 'hbgi', 'mean_mg_dl', 'coverage_pct']
FAMILIES = {
    'tir': {'tir_observed_pct': 1},
    'hypoglycemia': {k: -1 for k in ['tbr70_pct', 'tbr54_pct', 'lbgi',
                                    'prolonged_under54_120min_events',
                                    'hypo_events_per_observed_day']},
    'hyperglycemia': {k: -1 for k in ['tar180_pct', 'tar250_pct', 'hbgi']},
    'variability': {'sd_mg_dl': -1, 'cv_pct': -1},
}
BOOTSTRAP_SEED = 1030317  # Statistical resampling only; not a training seed.
BOOTSTRAP_REPLICATES = 20000
TOL = 1e-8  # Old joint decision tolerance; never used for score verification.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def job_key(job):
    return '%s_p%02d_s%d' % (job['group'], job['patient'], job['seed'])


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def exact(actual, saved, path):
    """Point to the first mismatched field; do not relax numeric equality."""
    if isinstance(actual, dict) and isinstance(saved, dict):
        if actual.keys() != saved.keys():
            raise ValueError(path + ': different dictionary keys')
        for key in actual:
            exact(actual[key], saved[key], path + '.' + key)
    elif isinstance(actual, list) and isinstance(saved, list):
        if len(actual) != len(saved):
            raise ValueError(path + ': different list lengths')
        for i, (a, b) in enumerate(zip(actual, saved)):
            exact(a, b, '%s[%d]' % (path, i))
    elif actual != saved:
        raise ValueError('%s: recomputed=%r, saved=%r; use the original scoring '
                         'runtime if floating-point versions differ' % (path, actual, saved))


def stats(values):
    present = [v for v in values if v is not None]
    return dict(mean=float(np.mean(present)) if present else None,
                sd=float(np.std(present, ddof=1)) if len(present) > 1 else None,
                n_patients=len(present))


def patient_tables(cases):
    by_patient = {}
    for case in cases:
        by_patient.setdefault(str(case['job']['patient']), []).append(case)
    patients = {}
    for pid, entries in sorted(by_patient.items(), key=lambda x: int(x[0])):
        patient = dict(case_count=len(entries), keys=[c['key'] for c in entries])
        for domain, names in [('bg', GLUCOSE), ('cgm', GLUCOSE), ('dose', DOSE)]:
            means = {}; counts = {}
            for name in names:
                values = [(c['metrics'] if domain == 'dose' else c['metrics'][domain]).get(name)
                          for c in entries]
                present = [v for v in values if v is not None]
                means[name] = float(np.mean(present)) if present else None
                counts[name] = len(present)
            patient[domain] = means
            patient[domain+'_case_counts'] = counts
        patients[pid] = patient
    table = {domain: {k: stats([p[domain][k] for p in patients.values()]) for k in names}
             for domain, names in [('bg', GLUCOSE), ('cgm', GLUCOSE), ('dose', DOSE)]}
    return patients, table


def load_panel(folder, name):
    summary_path = folder/'summary.json'; manifest_path = folder/'manifest.json'
    summary = read_json(summary_path); manifest = read_json(manifest_path)
    protocol = manifest['protocol']
    if protocol['scorer'] != str(SCORER.relative_to(P)) or sha(SCORER) != protocol['scorer_sha256']:
        raise ValueError(name + ': frozen scorer path/hash mismatch')
    if manifest.get('scorer_sha256', sha(SCORER)) != sha(SCORER):
        raise ValueError(name + ': manifest scorer hash mismatch')
    if protocol['warmup_minutes'] != 360 or (protocol['total_minutes']-360) % 5:
        raise ValueError(name + ': unsupported scoring schedule')
    split = manifest.get('split', manifest.get('config', {}).get('split'))
    smoke = manifest.get('smoke', manifest.get('config', {}).get('smoke', False))
    splits = ('train', 'world_validation', 'development', 'confirmation')
    if split not in splits or ('split' in summary and summary['split'] != split):
        raise ValueError(name + ': split not identified or inconsistent')
    seed_sets = {s: set(protocol[s+'_scenario_seeds']) for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            if seed_sets[a] & seed_sets[b]:
                raise ValueError(name + ': scenario split overlap')
    jobs = manifest['jobs']; expected = {job_key(j): j for j in jobs}
    if len(expected) != len(jobs) or not jobs:
        raise ValueError(name + ': empty or duplicate manifest jobs')
    expected_cells = {(p, s, f) for p in protocol['patients']
                      for s in seed_sets[split] for f in protocol['bolus_factors']}
    cells = {(j['patient'], j['seed'], j['bolus_factor']) for j in jobs}
    if not cells <= expected_cells or len(cells) != len(jobs) or (not smoke and cells != expected_cells):
        raise ValueError(name + ': job cells do not match the frozen split')
    for obj in (manifest, summary):
        if 'jobs_sha256' in obj and obj['jobs_sha256'] != json_sha(jobs):
            raise ValueError(name + ': jobs hash mismatch')
    entries = summary['episodes']; keys = [e['key'] for e in entries]
    if len(set(keys)) != len(keys) or set(keys) != set(expected) or summary['count'] != len(keys):
        raise ValueError(name + ': missing/duplicate jobs; no incomplete subset comparison')
    cases = []
    for entry in sorted(entries, key=lambda e: e['key']):
        key = entry['key']; job = expected[key]
        path = (folder/entry.get('raw_path', key+'.json')).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError(name + ': raw path escapes result directory')
        raw = read_json(path); raw_sha = sha(path)
        if 'raw_sha256' in entry and raw_sha != entry['raw_sha256']:
            raise ValueError(name + '/' + key + ': raw hash mismatch')
        exact(job, raw['job'], name+'/'+key+'.job')
        for field in ('patient', 'seed', 'group', 'bolus_factor'):
            if field in entry:
                exact(job[field], entry[field], name+'/'+key+'.'+field)
        exact(raw['failure_reason'], entry['failure_reason'], name+'/'+key+'.failure_reason')
        records = raw['records']; planned = (job['total_minutes']-360)//5
        if job['total_minutes'] != protocol['total_minutes'] or planned <= 0:
            raise ValueError(name + '/' + key + ': job duration differs from protocol')
        # Every saved row is the end of the preceding five-minute interval.
        for i, record in enumerate(records):
            if record['minute'] != (i+1)*5 or record['warmup'] != (i < 72):
                raise ValueError(name + '/' + key + ': noncontiguous or shifted record timing')
        if len(records) > job['total_minutes']//5:
            raise ValueError(name + '/' + key + ': more records than scheduled')
        metrics = summarize(records, planned, raw['failure_reason'] is not None)
        exact(metrics, raw['metrics'], name+'/'+key+'.raw.metrics')
        exact(metrics, entry['metrics'], name+'/'+key+'.summary.metrics')
        provenance = raw.get('evaluation_provenance')
        if provenance:
            for field, value in [('manifest_sha256', sha(manifest_path)),
                                 ('jobs_sha256', json_sha(jobs)), ('scorer_sha256', sha(SCORER))]:
                exact(value, provenance[field], name+'/'+key+'.provenance.'+field)
        rows = metrics['observed_intervals']
        unknown = planned-rows
        complete = raw['failure_reason'] is None and rows == planned and all(
            metrics[signal]['coverage_pct'] == 100 for signal in ('bg', 'cgm'))
        cases.append(dict(
            key=key, job=job, raw_path=str(path.relative_to(P)), raw_sha256=raw_sha,
            failure_reason=raw['failure_reason'], technical_failure_reason=raw.get('technical_failure_reason'),
            raw_trajectory_available=raw.get('raw_trajectory_available', True),
            complete_followup=complete, observed_record_intervals=rows,
            unknown_tail_intervals=unknown, unknown_tail_minutes=unknown*5,
            unknown_tail=unknown > 0,
            unobserved_or_invalid_intervals={s: planned-metrics[s]['observed_intervals']
                                            for s in ('bg', 'cgm')},
            metrics=metrics))
    patients, observed = patient_tables(cases)
    # A full trajectory saved during a failed run is still retained, while the
    # method cannot enter a complete-run comparison until the run is completed.
    complete = summary['status'] == 'completed' and all(c['complete_followup'] for c in cases)
    return dict(
        name=name, folder=str(folder.relative_to(P)), split=split,
        smoke=smoke,
        controller=manifest.get('kind', manifest.get('config', {}).get('controller')),
        status=summary['status'], error=summary.get('error'),
        manifest_sha256=sha(manifest_path), summary_sha256=sha(summary_path),
        scoring_verified_exact=True, scorer_sha256=sha(SCORER),
        jobs_sha256=json_sha(jobs), jobs=sorted(jobs, key=job_key),
        action_support=manifest.get('action_support', '0..min(20,2*observed anchor) U/h'),
        checkpoint_sha256=manifest.get('checkpoint_sha256'),
        episodes=len(cases), patient_count=len(patients),
        failed_episodes=sum(c['metrics']['failed'] for c in cases),
        incomplete_episodes=sum(not c['complete_followup'] for c in cases),
        unknown_tail_intervals=sum(c['unknown_tail_intervals'] for c in cases),
        eligible_for_complete_followup_comparison=complete,
        table_label='Complete' if complete else 'Observed — incomplete/failed; no complete-method ranking',
        complete=observed if complete else None, observed=observed,
        coverage_and_tir_bounds={s: {k: observed[s][k] for k in
                                  ('coverage_pct', 'tir_lower_bound_pct', 'tir_upper_bound_pct')}
                                 for s in ('bg', 'cgm')},
        observed_event_totals={s: {
            'prolonged_under54_120min_events': sum(c['metrics'][s].get('prolonged_under54_120min_events', 0) for c in cases),
            'hypo_events': sum(len(c['metrics'][s].get('hypo_events', [])) for c in cases),
            'unknown_tail_may_contain_additional_events': any(c['unobserved_or_invalid_intervals'][s] for c in cases),
        } for s in ('bg', 'cgm')},
        patients=patients, cases=cases)


def paired_differences(candidate, reference, bootstrap):
    ids = sorted(candidate['patients'], key=int)
    if ids != sorted(reference['patients'], key=int):
        raise ValueError('Different patient clusters')
    samples = (np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(ids), (BOOTSTRAP_REPLICATES, len(ids)))
               if bootstrap and len(ids) > 1 else None)
    out = {}
    for domain, names in [('bg', GLUCOSE), ('cgm', GLUCOSE), ('dose', DOSE)]:
        out[domain] = {}
        for metric in names:
            deltas = {}
            for pid in ids:
                a = candidate['patients'][pid][domain][metric]
                b = reference['patients'][pid][domain][metric]
                if a is not None and b is not None:
                    deltas[pid] = a-b
            result = stats(list(deltas.values()))
            result.update(patient_differences=deltas, missing_patient_ids=[i for i in ids if i not in deltas],
                          patient_cluster_bootstrap_ci95=None)
            if samples is not None and len(deltas) == len(ids):
                boot = np.asarray(list(deltas.values()))[samples].mean(axis=1)
                result['patient_cluster_bootstrap_ci95'] = [float(x) for x in np.quantile(boot, [.025, .975])]
            out[domain][metric] = result
    return out


def original_joint(candidate, reference):
    changes = {k: candidate['complete']['bg'][k]['mean']-reference['complete']['bg'][k]['mean']
               for k in OLD_METRICS}
    gates = dict(tir_not_lower=changes['tir_lower_bound_pct'] >= -TOL,
                 complete=candidate['failed_episodes'] == 0 and candidate['complete']['bg']['coverage_pct']['mean'] == 100.,
                 hypo_strict=min(changes[k] for k in ('tbr70_pct', 'tbr54_pct')) < -TOL)
    gates.update({k: changes[k] <= TOL for k in
                  ['tbr70_pct', 'tbr54_pct', 'tar180_pct', 'tar250_pct', 'hbgi', 'sd_mg_dl', 'cv_pct']})
    return dict(applicable=True, passed=all(gates.values()), gates=gates, changes=changes)


def family_comparison(differences):
    families = {}; votes = dict(improved=0, tied=0, worsened=0)
    for family, directions in FAMILIES.items():
        metrics = {}
        for name, direction in directions.items():
            delta = differences['bg'][name]['mean']
            vote = 'improved' if delta*direction > TOL else 'worsened' if delta*direction < -TOL else 'tied'
            metrics[name] = dict(candidate_minus_reference=delta, direction=direction, vote=vote)
            votes[vote] += 1
        present = {m['vote'] for m in metrics.values()}
        status = ('tradeoff' if {'improved', 'worsened'} <= present else
                  'improved' if 'improved' in present else 'worsened' if 'worsened' in present else 'tied')
        families[family] = dict(status=status, metrics=metrics)
    return dict(descriptive_only=True, families=families, metric_votes=votes,
                family_votes={s: sum(f['status'] == s for f in families.values())
                              for s in ('improved', 'tied', 'worsened', 'tradeoff')},
                strict_majority_of_metrics_improved=votes['improved'] > sum(votes.values())/2,
                strict_majority_of_families_improved=sum(f['status'] == 'improved' for f in families.values()) > len(families)/2,
                no_SOTA_claim=True)


def compare(candidate, reference):
    for field in ('jobs', 'split', 'smoke'):
        exact(candidate[field], reference[field], candidate['name']+'.matching.'+field)
    complete = all(p['eligible_for_complete_followup_comparison'] for p in (candidate, reference))
    descriptive = paired_differences(candidate, reference, bootstrap=False)
    out = dict(candidate=candidate['name'], reference=reference['name'],
               difference_direction='candidate minus reference', identical_jobs_meals_factors=True,
               complete_pair_applicable=complete, observed_patient_differences=descriptive,
               observed_warning=None if complete else 'Available-prefix patient means may cover different intervals and case subsets; no complete-method comparison or bootstrap.',
               original_joint_nonregression=dict(applicable=False, passed=None),
               metric_family_comparison=None, complete_patient_differences=None)
    if complete:
        paired = paired_differences(candidate, reference, bootstrap=True)
        out.update(complete_patient_differences=paired,
                   original_joint_nonregression=original_joint(candidate, reference),
                   metric_family_comparison=family_comparison(paired))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', action='append', required=True, help='Directory name under results; repeat per method')
    ap.add_argument('--reference', help='One of the panel names; defaults to the first panel')
    ap.add_argument('--output', type=Path, required=True, help='New checks/panels_*.json path relative to this script')
    args = ap.parse_args()
    if len(set(args.panel)) != len(args.panel) or any(Path(n).name != n or n in ('.', '..') for n in args.panel):
        raise ValueError('Panel names must be unique result directory names')
    reference = args.reference or args.panel[0]
    if reference not in args.panel:
        raise ValueError('--reference must also appear in --panel')
    output = (R/args.output).resolve()
    if args.output.is_absolute() or output.parent != (R/'checks').resolve() or not output.name.startswith('panels_') or output.suffix != '.json':
        raise ValueError('--output must be checks/panels_*.json relative to the new research directory')
    if output.exists():
        raise FileExistsError('Refusing to replace an existing report: '+str(output))
    panels = {name: load_panel(R/'results'/name, name) for name in args.panel}
    comparisons = {name: compare(panel, panels[reference]) for name, panel in panels.items() if name != reference}
    payload = dict(
        schema=1, reference=reference, panel_order=args.panel, panels=panels, comparisons=comparisons,
        methodology=dict(
            bg_primary=True, cgm_secondary=True, scoring_dt_minutes=5,
            scorer=str(SCORER.relative_to(P)), scorer_sha256=sha(SCORER),
            raw_and_summary_score_verification='Every field exactly equal to unchanged control_metrics.summarize; no numeric tolerance',
            trajectory_glucose_sd_ddof=0, between_patient_sd_ddof=1,
            aggregation='First equal-weight case means within each patient, then equal-weight patient mean and sample SD; a missing metric is null, never zero. Observed tables retain available-case counts.',
            tir_bounds='Bounds for unknown/invalid outcomes, not confidence intervals',
            incomplete_rule='Every planned case is retained. Any failure, invalid signal, unknown tail or failed run excludes the entire panel from complete-followup comparisons; no complete-case subset ranking.',
            hypoglycemia_event_definition='Original scorer: >=15 minutes <70; closes after >=15 minutes recovery. Long54: continuous <54 for >=120 minutes. Events are observed, possibly right-censored.',
            old_joint_source='RL_DSENet_公平低糖_2026-09-21/aggregate.py:18; unchanged BG gates, applied only to two complete panels',
            aggregation_source='RL_DSENet_公平低糖_2026-09-21/aggregate.py and final_analysis.py',
            family_rule='BG only. A family improves when every component is non-worse and at least one improves; mixed directions are tradeoff. Component vote counts are descriptive and distinct from the original joint criterion.',
            decision_tolerance=TOL, bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                unit='paired patient cluster after within-patient averaging', seed_is_training_seed=False,
                interval='percentile 95%; unavailable for one patient or incomplete followup',
                descriptive_no_multiplicity_correction=True),
            limitations=['Single training seed; patient variability is not seed variability',
                         'Known virtual patients; no unseen-patient or clinical claim',
                         'Different action support is disclosed per panel; comparison does not isolate architecture',
                         'Train/validation/development/confirmation panels cannot be mixed in one comparison',
                         'No automatic SOTA claim or candidate selection']),
        provenance=dict(script_sha256=sha(Path(__file__)), numpy_version=np.__version__,
                        python_version=sys.version, protocol_sha256=sha(R/'protocol.json')))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='all_raw_scores_exact', output=str(output), panels=len(panels),
                          episodes=sum(p['episodes'] for p in panels.values()),
                          complete_panels=[n for n, p in panels.items() if p['complete'] is not None]), ensure_ascii=False))


if __name__ == '__main__':
    main()
