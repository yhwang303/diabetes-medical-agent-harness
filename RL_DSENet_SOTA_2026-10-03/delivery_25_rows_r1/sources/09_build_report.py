"""Render audited JSON evidence to an offline Chinese research report.

Standard library only. Never imports a scorer/model, reads raw trajectories,
selects a checkpoint, or changes training/evaluation evidence. Input panel
scores must already have passed summarize_panels' exact original-score audit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

R = Path(__file__).resolve().parent
P = R.parent
SCORER_SHA = '0a6d44298d942bc09764f15e7ef45a71d3afbc7ba63cad3dae5425f0b22b8cfd'
READ_HASHES = {}
PRIMARY = [
    ('tir_observed_pct', 'TIR 70–180 % ↑', 1), ('tbr70_pct', 'TBR <70 % ↓', -1),
    ('tbr54_pct', 'TBR <54 % ↓', -1), ('tar180_pct', 'TAR >180 % ↓', -1),
    ('tar250_pct', 'TAR >250 % ↓', -1), ('sd_mg_dl', '血糖 SD mg/dL ↓', -1),
    ('cv_pct', 'CV % ↓', -1), ('lbgi', 'LBGI ↓', -1), ('hbgi', 'HBGI ↓', -1),
]
EXTRA = [('bg', 'mean_mg_dl', 'Mean BG mg/dL'),
         ('dose', 'basal_u_per_observed_day', '基础输注 U/观测日'),
         ('dose', 'bolus_u_per_observed_day', '外源 bolus U/观测日'),
         ('dose', 'action_tv_per_observed_day', '动作 TV (U/h)/观测日'),
         ('bg', 'hypo_events_per_observed_day', '低糖事件/观测日')]
GROUPS = {'comparison': '系统对比', 'internal_ablation': '内部消融'}
SPLITS = {'development': '开发场景', 'confirmation': '独立确认场景',
          'world_validation': 'World 开发验证场景'}
esc = html.escape


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def parse_json(text):
    # Duplicate keys and NaN/Infinity can silently invalidate audit evidence.
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    path = Path(path).resolve(); data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(path not in READ_HASHES or READ_HASHES[path] == digest, 'JSON changed during build: ' + str(path))
    READ_HASHES[path] = digest
    return parse_json(data.decode('utf-8'))


def checked_path(value, panels=False):
    path = Path(value)
    if not path.is_absolute():
        path = R / path
    path = path.resolve()
    path.relative_to((R / 'checks').resolve())
    require(path.is_file() and path.suffix == '.json', 'Expected an existing checks JSON')
    if panels:
        require(path.name.startswith('panels_'), 'Panels must be audited checks/panels_*.json')
    return path


def number(value, nullable=False):
    require((nullable and value is None) or
            (type(value) in (int, float) and math.isfinite(value)), 'Expected a finite JSON number')
    return value


def same_number(actual, expected, label):
    # Only checks aggregate JSON self-consistency across NumPy/stdlib summation.
    # This is NOT a relaxation of the original raw-score exact equality audit.
    require((actual is None and expected is None) or
            (actual is not None and expected is not None and
             math.isclose(number(actual), number(expected), rel_tol=1e-10, abs_tol=1e-9)),
            'Aggregate inconsistency: ' + label)


def job_key(job):
    return '%s_p%02d_s%d' % (job['group'], job['patient'], job['seed'])


def validate_panel(panel, protocol):
    require(panel['scoring_verified_exact'] is True and panel['scorer_sha256'] == SCORER_SHA,
            'Panel lacks exact original-score audit')
    for field in ('manifest_sha256', 'summary_sha256', 'jobs_sha256'):
        hash_value(panel[field])
    if panel['checkpoint_sha256'] is not None:
        hash_value(panel['checkpoint_sha256'])
    require(type(panel['smoke']) is bool, 'Panel smoke flag must be explicit')
    split = panel['split']
    require(split in SPLITS, 'Training data cannot enter a report comparison')
    jobs = panel['jobs']
    require(len(jobs) == panel['episodes'] > 0, 'Episode/job count mismatch')
    cells = {(j['patient'], j['seed'], j['bolus_factor']) for j in jobs}
    expected = {(p, s, f) for p in protocol['patients']
                for s in protocol[split + '_scenario_seeds'] for f in protocol['bolus_factors']}
    require(len(cells) == len(jobs) and cells <= expected and (panel['smoke'] or cells == expected),
            'Incomplete or wrong split job matrix')
    for j in jobs:
        require(j['total_minutes'] == protocol['total_minutes'] and isinstance(j['meals'], list),
                'Episode duration/meal schema mismatch')
        require(j['group'] == 'bolus_%.1f' % j['bolus_factor'], 'Job group/factor mismatch')
        require(all(len(m) == 2 and number(m[0]) >= 0 and number(m[1]) >= 0 for m in j['meals']),
                'Invalid meal schedule')
    expected_jobs = {job_key(j): j for j in jobs}
    cases = panel['cases']
    require(len(cases) == len(jobs) and len({c['key'] for c in cases}) == len(cases), 'Case matrix mismatch')
    planned = (protocol['total_minutes'] - protocol['warmup_minutes']) // 5
    for c in cases:
        require(c['key'] in expected_jobs and c['job'] == expected_jobs[c['key']], 'Case job/meal mismatch')
        hash_value(c['raw_sha256'])
        metrics = c['metrics']
        require(type(metrics['failed']) is bool and metrics['failed'] == (c['failure_reason'] is not None),
                'Case failure evidence mismatch')
        require(metrics['planned_intervals'] == planned and
                0 <= metrics['observed_intervals'] <= planned, 'Invalid planned/observed interval count')
        unknown = planned - metrics['observed_intervals']
        require(c['observed_record_intervals'] == metrics['observed_intervals'] and
                c['unknown_tail_intervals'] == unknown and c['unknown_tail_minutes'] == unknown * 5 and
                c['unknown_tail'] == (unknown > 0), 'Unknown-tail mismatch')
        for signal in ('bg', 'cgm'):
            sm = metrics[signal]
            require(c['unobserved_or_invalid_intervals'][signal] == planned - sm['observed_intervals'],
                    'Signal coverage count mismatch')
            same_number(sm['coverage_pct'], 100 * sm['observed_intervals'] / planned, 'case coverage')
        complete = not metrics['failed'] and unknown == 0 and all(
            metrics[d]['coverage_pct'] == 100 for d in ('bg', 'cgm'))
        require(type(c['complete_followup']) is bool and c['complete_followup'] == complete,
                'Case falsely labelled complete')
    require(panel['failed_episodes'] == sum(c['metrics']['failed'] for c in cases) and
            panel['incomplete_episodes'] == sum(not c['complete_followup'] for c in cases) and
            panel['unknown_tail_intervals'] == sum(c['unknown_tail_intervals'] for c in cases),
            'Panel failure/coverage counts disagree')
    complete = panel['status'] == 'completed' and all(c['complete_followup'] for c in cases)
    require(type(panel['eligible_for_complete_followup_comparison']) is bool and
            panel['eligible_for_complete_followup_comparison'] == complete, 'Panel completion mismatch')
    require(panel['complete'] == (panel['observed'] if complete else None), 'Observed panel promoted to complete')
    patients = panel['patients']
    require(set(patients) == {str(j['patient']) for j in jobs} and len(patients) == panel['patient_count'],
            'Patient count mismatch')
    for pid, patient in patients.items():
        entries = [c for c in cases if str(c['job']['patient']) == pid]
        require(patient['case_count'] == len(entries) and set(patient['keys']) == {c['key'] for c in entries},
                'Patient cases mismatch')
        for domain, table in panel['observed'].items():
            require(domain in ('bg', 'cgm', 'dose'), 'Unknown score domain')
            for metric in table:
                values = [(c['metrics'] if domain == 'dose' else c['metrics'][domain]).get(metric) for c in entries]
                values = [number(x) for x in values if x is not None]
                require(patient[domain + '_case_counts'][metric] == len(values), 'Missing-case count mismatch')
                same_number(patient[domain][metric], statistics.fmean(values) if values else None, metric)
    for domain, table in panel['observed'].items():
        for metric, stat in table.items():
            require(set(stat) == {'mean', 'sd', 'n_patients'}, 'Unknown aggregate schema')
            values = [number(p[domain][metric]) for p in patients.values() if p[domain][metric] is not None]
            require(stat['n_patients'] == len(values), 'Aggregate sample size mismatch')
            same_number(stat['mean'], statistics.fmean(values) if values else None, metric)
            same_number(stat['sd'], statistics.stdev(values) if len(values) > 1 else None, metric + '.sd')
    for signal in ('bg', 'cgm'):
        for metric in ('coverage_pct', 'tir_lower_bound_pct', 'tir_upper_bound_pct'):
            require(panel['coverage_and_tir_bounds'][signal][metric] == panel['observed'][signal][metric],
                    'Coverage/TIR bounds mismatch')
        totals = panel['observed_event_totals'][signal]
        require(totals['prolonged_under54_120min_events'] == sum(
            c['metrics'][signal].get('prolonged_under54_120min_events', 0) for c in cases), 'Long event count mismatch')
    return panel


def load_panels(path, protocol, confirmation=False):
    data = read_json(path)
    require(data['schema'] == 1, 'Unsupported panels schema')
    method = data['methodology']
    require(method['scorer_sha256'] == SCORER_SHA and method['bg_primary'] is True and
            method['cgm_secondary'] is True and method['scoring_dt_minutes'] == 5 and
            method['between_patient_sd_ddof'] == 1 and method['trajectory_glucose_sd_ddof'] == 0,
            'Scorer or statistical contract mismatch')
    require(data['provenance']['protocol_sha256'] == sha(R / 'protocol.json') and
            data['provenance']['script_sha256'] == sha(R / 'summarize_panels.py'), 'Panel audit provenance mismatch')
    order = data['panel_order']
    require(len(set(order)) == len(order) and set(order) == set(data['panels']) and data['reference'] in order,
            'Panel order/reference mismatch')
    panels = [validate_panel(data['panels'][n], protocol) for n in order]
    require(all(p['name'] == n for p, n in zip(panels, order)), 'Panel name mismatch')
    first = panels[0]
    require(all(p['split'] == first['split'] and p['smoke'] == first['smoke'] and
                p['jobs'] == first['jobs'] and p['jobs_sha256'] == first['jobs_sha256'] for p in panels),
            'Methods must have exactly aligned jobs, meals, factors and split')
    require((first['split'] == 'confirmation') == confirmation, 'Confirmation input must use --confirmation explicitly')
    if confirmation:
        from summarize_confirmation import validate_bundle
        validate_bundle(data)
    require(set(data['comparisons']) == set(order) - {data['reference']}, 'Missing paired comparison audit')
    for name, comparison in data['comparisons'].items():
        require(comparison['candidate'] == name and comparison['reference'] == data['reference'] and
                comparison['identical_jobs_meals_factors'] is True, 'Paired comparison identity mismatch')
        a, b = data['panels'][name], data['panels'][data['reference']]
        complete = a['eligible_for_complete_followup_comparison'] and b['eligible_for_complete_followup_comparison']
        require(comparison['complete_pair_applicable'] == complete and
                (comparison['complete_patient_differences'] is not None) == complete,
                'Incomplete comparison falsely supplies full-followup inference')
        differences = comparison['complete_patient_differences'] if complete else comparison['observed_patient_differences']
        for domain, scores in differences.items():
            for metric, entry in scores.items():
                for pid, difference in entry['patient_differences'].items():
                    same_number(difference, a['patients'][pid][domain][metric] - b['patients'][pid][domain][metric],
                                'paired.' + metric)
                vals = list(entry['patient_differences'].values())
                same_number(entry['mean'], statistics.fmean(vals) if vals else None, 'paired mean')
                ci = entry['patient_cluster_bootstrap_ci95']
                require(ci is None or (complete and len(ci) == 2 and number(ci[0]) <= number(ci[1])),
                        'Invalid paired confidence interval')
    return data


def validate_world(diag, binding, protocol):
    require(diag['schema'] == binding['schema'] == 1 and
            diag['evidence_kind'] == 'real_saved_world_validation_predictions' and
            binding['evidence_kind'] == 'full_saved_prediction_recomputation' and
            binding['status'] == 'passed' and binding['passed'] is True and
            binding['split'] == 'world_validation' and binding['sampled_comparison'] is False and
            binding['all_natural_and_paired_rows_required'] is True, 'World lacks full prediction identity audit')
    require(diag['step'] == binding['step'] and diag['training_seed'] == protocol['training_seed'] and
            diag['protocol_sha256'] == sha(R / 'protocol.json') and
            diag['provenance_sha256'] == binding['world_dependency']['training_provenance_sha256'] and
            diag['completion_sha256'] == binding['world_dependency']['completion_sha256'], 'World identity mismatch')
    budget = diag['training_budget']; bound_budget = binding['training_budget']
    require(budget['completed_steps'] == bound_budget['completed_steps'] == budget['configured_steps'] ==
            bound_budget['configured_steps'] and budget['smoke_budget_override'] is False and
            bound_budget['budget_override'] is False, 'World formal training budget is incomplete or overridden')
    for kind in ('natural', 'paired'):
        source = diag['inputs'][kind]
        require(source['prediction_sha256'] == binding['datasets'][kind]['prediction_sha256'] and
                source['labels']['manifest_sha256'] == binding['datasets'][kind]['manifest_sha256'],
                'World prediction/label hash mismatch')
        order = source['labels']['file_order']; bound_order = binding['datasets'][kind]['file_order']
        require(len(order) == len(bound_order) and all(
            all(a[k] == b[k] for k in ('file', 'sha256', 'job')) and a['origins'] == b['count']
            for a, b in zip(order, bound_order)), 'World label row order mismatch')
        cells = {(x['job']['patient'], x['job']['seed'], x['job']['bolus_factor']) for x in order}
        require(cells == {(p, s, f) for p in protocol['patients'] for s in protocol['world_validation_scenario_seeds']
                          for f in protocol['bolus_factors']} and len(cells) == len(order), 'World validation job matrix mismatch')
        require(all(v['passed'] is True for v in diag['trainer_metric_reproduction'][kind]), 'World metric audit failed')
        comparison = binding['comparisons'][kind]
        require(comparison['passed'] is True and comparison['sample_order_exact'] is True, 'World row order audit failed')
        for field in ('cgm_quantiles', 'bg_three_class_probabilities'):
            require(comparison[field]['all_elements_checked'] is True and comparison[field]['passed'] is True and
                    comparison[field]['compared_elements'] == comparison[field]['expected_elements'],
                    'World binding is only partial')
        for t in ('70', '54'):
            event = diag['metrics'][kind]['bg_horizon_events'][t]
            require(event['positive'] + event['negative'] == event['complete_windows'] and
                    event['true_positives'] + event['false_negatives'] == event['positive'] and
                    event['threshold'] == .5, 'World event denominator mismatch')
            same_number(event['fnr'], event['false_negatives'] / event['positive'] if event['positive'] else None,
                        'world FNR; zero positives must be null')
    return diag


def validate_actions(data, panels):
    require(data['schema'] == 1 and data['panel'] in panels and
            data['status'] == 'all_actions_linked_to_actual_requests' and data['efficacy_or_causal_claim'] is False,
            'Action diagnostic lacks actual-request linkage')
    panel = panels[data['panel']]
    require(data['manifest_sha256'] == panel['manifest_sha256'] and data['summary_sha256'] == panel['summary_sha256'] and
            data['split'] == panel['split'] and data['smoke'] == panel['smoke'], 'Action/panel identity mismatch')
    count = sum(c['observed_record_intervals'] for c in panel['cases'])
    require(data['linked_decisions'] == data['overall']['decisions'] == count and
            set(data['by_case']) == {c['key'] for c in panel['cases']}, 'Action count/cases mismatch')
    for c in panel['cases']:
        require(data['by_case'][c['key']]['decisions'] == c['observed_record_intervals'], 'Case action count mismatch')
    require(sum(x['decisions'] for x in data['by_predecision_cgm'].values()) == count and
            data['max_request_absolute_error_u_h'] == 0, 'Action linkage/bin count mismatch')
    require(sum(data['overall'][key] for key in ('at_anchor_count', 'below_anchor_count', 'above_anchor_count')) == count,
            'Action anchor count mismatch')
    return data


def validate_summary(summary, panels, registry, status, confirmation_paths):
    require(summary['schema'] == 1 and isinstance(summary['paragraphs'], list) and
            all(isinstance(x, str) and x.strip() for x in summary['paragraphs']), 'Invalid reviewable report summary')
    require(set(summary['panels']) == set(panels), 'Every score panel needs an explicit method/group label')
    methods = {m['id']: m for m in registry['methods']}
    require(len(methods) == len(registry['methods']), 'Duplicate registry method ID')
    for name, spec in summary['panels'].items():
        require(spec['method_id'] in methods and spec['group'] in GROUPS and
                isinstance(spec['label'], str) and spec['label'].strip(), 'Invalid panel method definition')
        method = methods[spec['method_id']]
        kind=panels[name]['controller'];underlying={'ppo_mean':'ppo','world_ppo_mean':'world_ppo'}.get(kind,kind)
        require(method.get('entrypoint', {}).get('kind') == underlying, 'Panel/registry algorithm mismatch')
        if kind!=underlying:
            require(spec['group']=='internal_ablation','Mean deployment belongs to the original algorithm internal ablation')
            if panels[name]['split']=='confirmation':
                deployment=panels[name]['deployment']
                require(deployment['underlying_policy_kind']==underlying and
                        deployment['deployment_rule']=='probability_mean_of_capped_executable_grid' and
                        deployment['same_checkpoint_new_deployment_ablation'] is True and
                        deployment['original_preregistration'] is False and deployment['new_external_method'] is False,
                        'Mean confirmation deployment evidence mismatch')
        if spec['group'] == 'internal_ablation':
            require(isinstance(spec.get('deployment_variant'), str) and spec['deployment_variant'].strip(),
                    'Internal deployment ablation must state its variant under the original algorithm')
        if method.get('checkpoint', {}).get('sha256'):
            require(method['checkpoint']['sha256'] == panels[name]['checkpoint_sha256'], 'Frozen method checkpoint mismatch')
    for claim in summary.get('equal_bg', []):
        a, b = (panels[claim[k]] for k in ('left', 'right'))
        require(a['jobs'] == b['jobs'] and a['complete'] is not None and b['complete'] is not None and
                a['complete']['bg'] == b['complete']['bg'] and
                all(a['patients'][pid]['bg'] == b['patients'][pid]['bg'] for pid in a['patients']),
                'Requested BG equality claim is not supported by aggregate and patient scores')
    if status == 'final':
        require(confirmation_paths, 'Final report requires explicit --confirmation evidence')
        confirm = {n: p for n, p in panels.items() if p['split'] == 'confirmation'}
        require(confirm and all(not p['smoke'] and p['scoring_verified_exact'] for p in confirm.values()),
                'Final confirmation needs audited nonsmoke score evidence for every method')
        review = summary.get('confirmation_review', {})
        require(review.get('reviewed') is True and isinstance(review.get('reviewer'), str) and review['reviewer'].strip(),
                'Final status needs an explicit review record; it never promotes a model')
        require(set(review.get('panel_manifest_sha256', {})) == set(confirm) and
                all(review['panel_manifest_sha256'][n] == p['manifest_sha256'] for n, p in confirm.items()),
                'Confirmation review must bind every reported confirmation panel manifest')
    return methods


def fmt(value, places=2):
    return '—' if value is None else format(number(value), '.' + str(places) + 'f')


def stat_text(stat):
    if stat['mean'] is None:
        return '—（无观测）'
    return fmt(stat['mean']) + (' ± ' + fmt(stat['sd']) if stat['sd'] is not None else '（SD不可估）')


def table(headers, rows, classes=None):
    body = []
    for i, row in enumerate(rows):
        body.append('<tr' + (' class="' + classes[i] + '"' if classes else '') + '>' +
                    ''.join('<th scope="row">' + x + '</th>' if j == 0 else '<td>' + x + '</td>'
                            for j, x in enumerate(row)) + '</tr>')
    return '<div class="table-scroll" tabindex="0"><table><thead><tr>' + ''.join(
        '<th scope="col">' + esc(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def score_table(panels, labels, signal, rows_csv, cells):
    complete = [p for p in panels if p['complete'] is not None and not p['smoke']]
    best = {}
    for key, _, direction in PRIMARY:
        values = [p['complete'][signal][key]['mean'] for p in complete if p['complete'][signal][key]['mean'] is not None]
        best[key] = (max(values) if direction > 0 else min(values)) if values else None
    rows = []
    for panel in panels:
        n = panel['name']; valid = panel['complete'] is not None
        scores = panel['observed'][signal]
        badge = 'Complete' if valid else 'Observed · 前缀/失败'
        if panel['smoke']:
            badge += ' · 工程烟测'
        row = [esc(labels[n]['label']) + '<small>' + esc(n) + '</small>', esc(badge)]
        for key, _, _ in PRIMARY:
            value = scores[key]; text = stat_text(value)
            winning = valid and not panel['smoke'] and best[key] is not None and value['mean'] == best[key]
            cell_id = signal + ':' + n + ':' + key
            cells[cell_id] = dict(text=text, best=winning, panel=n, metric=key, signal=signal)
            row.append('<span data-cell="' + esc(cell_id, quote=True) + '">' +
                       ('<b class="best">' + esc(text) + '</b>' if winning else esc(text)) + '</span>')
        low = scores['tir_lower_bound_pct']['mean']; high = scores['tir_upper_bound_pct']['mean']
        unobserved = sum(c['unobserved_or_invalid_intervals'][signal] for c in panel['cases']) * 5
        row += [str(panel['failed_episodes']) + '/' + str(panel['episodes']),
                esc(stat_text(scores['coverage_pct'])),
                str(panel['observed_event_totals'][signal]['prolonged_under54_120min_events']) + ('†' if not valid else ''),
                '[' + fmt(low) + ', ' + fmt(high) + ']', str(panel['unknown_tail_intervals'] * 5), str(unobserved)]
        rows.append(row)
        for domain in ('bg', 'cgm', 'dose'):
            if domain != signal:
                continue
            for key, value in panel['observed'][domain].items():
                rows_csv.append(dict(split=panel['split'],group=labels[n]['group'],panel=n,method_id=labels[n]['method_id'],
                    label=labels[n]['label'],signal=domain,metric=key,mean=value['mean'],sample_sd=value['sd'],
                    n_patients=value['n_patients'],complete=valid,smoke=panel['smoke'],failed_episodes=panel['failed_episodes'],
                    planned_episodes=panel['episodes'],unknown_tail_minutes=panel['unknown_tail_intervals']*5,
                    observed_long54_events=panel['observed_event_totals'][signal]['prolonged_under54_120min_events']))
    return '<p>可横向滚动查看其余指标；方法列固定。</p>' + table(['方法 / 部署版本', '随访状态'] + [x[1] for x in PRIMARY] +
                 ['失败病例', '覆盖 %', 'BG<54 ≥120min事件' if signal == 'bg' else 'CGM<54 ≥120min事件',
                  'TIR上下界 %', '未知尾分钟', '缺失/无效信号分钟'], rows,
                 ['complete' if p['complete'] is not None else 'observed' for p in panels])


def paired_table(bundle, labels, export):
    rows = []
    for name, c in bundle['comparisons'].items():
        complete = c['complete_pair_applicable']
        scores = c['complete_patient_differences'] if complete else c['observed_patient_differences']
        for key in ('tir_observed_pct', 'tbr70_pct', 'tbr54_pct', 'tar180_pct', 'sd_mg_dl'):
            stat = scores['bg'][key]; ci = stat['patient_cluster_bootstrap_ci95']
            row = [esc(labels[name]['label'] + ' − ' + labels[c['reference']]['label']), esc(key),
                   esc(stat_text(stat)), '[' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']' if ci else '—',
                   str(stat['n_patients']), '完整病例配对' if complete else 'Observed差值，不作完整比较']
            rows.append(row)
            export.append(dict(candidate=name,reference=c['reference'],metric=key,mean=stat['mean'],
                sample_sd=stat['sd'],ci95_low=ci[0] if ci else None,ci95_high=ci[1] if ci else None,
                n_patients=stat['n_patients'],complete_pair_applicable=complete))
    return table(['配对差：候选 − 参考', 'BG指标', '患者差 mean ± SD', '患者cluster bootstrap 95%CI', '患者数', '适用范围'], rows)


def world_section(worlds):
    rows = []; notes = []
    for diag, _ in worlds:
        for kind, title in (('natural', '自然行为轨迹'), ('paired', '同快照配对干预')):
            m = diag['metrics'][kind]; severe = m['bg_horizon_events']['54']
            low = m['bg_horizon_events']['70']; minimum = m['minimum_cgm']['complete_horizon']['bg_observed_below54']
            ranking = m.get('paired_response', {}).get('bg_event_order', {}).get('54')
            name = diag['variant'] + ' · step ' + str(diag['step'])
            rows.append([esc(name), title, str(severe['complete_windows']), str(severe['positive']),
                         str(severe['false_negatives']) + '/' + str(severe['positive']) if severe['positive'] else '不可评：无阳性',
                         fmt(100*severe['fnr']) + '%' if severe['fnr'] is not None else '—（不是0）',
                         fmt(100*low['fnr']) + '%' if low['fnr'] is not None else '—',
                         fmt(m['cgm_median']['mae_mg_dl']), fmt(minimum['bias_mg_dl']),
                         str(ranking['correct_pairs']) + '/' + str(ranking['eligible_pairs']) if ranking else '不适用'])
            if diag['variant'] == 'quantile' and kind == 'paired':
                notes.append('Quantile配对严重低糖：固定0.5概率阈值漏报 %d/%d（%s%%）；异臂排序 %d/%d。'
                    '两项衡量不同能力，排序好不等于绝对风险识别或最低值偏差已解决。' %
                    (severe['false_negatives'], severe['positive'], fmt(100*severe['fnr']) if severe['fnr'] is not None else '不可评',
                     ranking['correct_pairs'], ranking['eligible_pairs']))
    return '<section id="world"><h2>World：预测身份与低糖诊断</h2><div class="caution">' + ''.join(
        '<p>' + esc(n) + '</p>' for n in notes) + '</div>' + table(
        ['模型 / 选中步数', '验证分布', '完整窗口', 'BG<54阳性', 'BG54 FN/阳性', 'BG54 FNR', 'BG70 FNR',
         'CGM MAE mg/dL', '严重低糖子集最低CGM偏差', 'BG54异臂排序'], rows) + (
        '<p>自然数据无阳性时FNR不可评，零FN计数不能写成零漏报。事件指标使用真实BG；轨迹回归使用CGM。'
        '窗口重叠、同origin多臂及同患者场景并不独立。World validation用于选best，属于开发诊断；'
        '全量预测重算仅证明权重与保存预测一致。边际分位数不是联合轨迹分布或已校准CVaR。</p></section>')


CSS = '''
:root{--ink:#16332d;--muted:#586963;--line:#d8e0d9;--paper:#f3f3ea;--card:#fffef8;--accent:#23614c;--warn:#9d5727}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
header,main,footer{max-width:1800px;margin:auto}header{padding:48px 40px 28px;border-bottom:1px solid var(--line)}
.eyebrow{font:600 12px/1.4 monospace;letter-spacing:.15em;color:var(--accent)}h1{font-size:38px;line-height:1.25;margin:18px 0}
h2{font-size:24px;margin:0 0 14px}h3{font-size:18px;margin:24px 0 10px}main{padding:28px 32px}section{background:var(--card);border:1px solid var(--line);padding:26px;margin:0 0 24px}
.status{display:inline-block;background:#f5dfb9;color:#6c401d;padding:6px 12px;font-weight:650}.caution{border-left:4px solid var(--warn);padding:1px 18px;background:#fcf0da;margin:16px 0}
.lead{font-size:18px;max-width:1050px}nav{display:flex;gap:20px;flex-wrap:wrap;margin-top:24px}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}
.table-scroll{overflow:auto;max-height:620px;border:1px solid var(--line);margin:16px 0;isolation:isolate}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;background:var(--card)}thead th{position:sticky;top:0;z-index:3;background:#234b3f;color:#fff;font-weight:600}
tbody th:first-child,thead th:first-child{position:sticky;left:0;text-align:left;min-width:235px;max-width:330px;white-space:normal;border-right:1px solid var(--line)}tbody th:first-child{z-index:2}thead th:first-child{z-index:4}
tbody th{font-weight:500}tbody tr:nth-child(even) td,tbody tr:nth-child(even) th{background:#f3f6ef}.observed td,.observed th{background:#fff0dc!important}.best{color:#17563d;font-weight:750}small{display:block;color:var(--muted);font-size:11px;overflow-wrap:anywhere}
.mono{font:11px/1.6 ui-monospace,monospace;overflow-wrap:anywhere;word-break:break-all}.source-list li{padding:6px 0}.source-list code{font-size:11px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}footer{padding:0 40px 32px;color:var(--muted)}
@media(max-width:700px){header{padding:26px 18px}h1{font-size:28px}main{padding:16px 10px}section{padding:18px 12px}tbody th:first-child,thead th:first-child{min-width:170px;max-width:210px}}
@media print{@page{size:A3 landscape;margin:10mm}body{background:white}header,main,section{padding:10px}section{break-inside:avoid}.table-scroll{max-height:none;overflow:visible}th,td{font-size:8px;padding:4px}thead th,tbody th:first-child{position:static}nav{display:none}}
'''


def render(bundles, worlds, summary, methods, status, sources, score_csv, pair_csv, cells, actions=()):
    panels = {n: p for _, data in bundles for n, p in data['panels'].items()}
    labels = summary['panels']; confirmed = any(p['split'] == 'confirmation' for p in panels.values())
    state = ('研究进行中' if status == 'interim' else '研究报告（含已审核确认资料）') + (' · 已提供确认资料' if confirmed else ' · 确认未运行')
    content = ['<section id="reading"><h2>当前解读</h2>' + ''.join('<p class="lead">' + esc(s) + '</p>' for s in summary['paragraphs'])]
    for claim in summary.get('equal_bg', []):
        content.append('<p>' + esc(labels[claim['left']]['label'] + ' 与 ' + labels[claim['right']]['label']) +
                       '：完整BG汇总和逐患者BG均值逐字段一致，当前证据不支持前者优于后者。</p>')
    for action in actions:
        overall = action['overall']
        content.append('<p>' + esc(labels[action['panel']]['label']) + ' 的实际请求已逐个关联：%d 个决策中，%d 个在观测基础率 ±%s U/h 内。'
                       '分箱使用决策前CGM；这是部署动作诊断，不把argmax解释为已证实的因果机制。</p>' %
                       (action['linked_decisions'], overall['at_anchor_count'], format(action['at_anchor_tolerance_u_h'], '.8g')))
    content.append('<p>同一算法的不同检查点或部署版本可能占多行；应结合算法家族与部署定义识别，表格行数不等于独立算法数。'
                   '粗体仅描述本表完整方法的数值最优，相关指标的改善次数不等于独立胜出次数。</p>')
    content.append('<p><a href="报告解读.json">可审查解读输入</a>。本报告不自动选择模型、宣布晋升或给出普遍最优结论。</p></section>')
    for i, (_, bundle) in enumerate(bundles):
        first = bundle['panels'][bundle['panel_order'][0]]
        for group in GROUPS:
            chosen = [bundle['panels'][n] for n in bundle['panel_order'] if labels[n]['group'] == group]
            if not chosen:
                continue
            title = SPLITS[first['split']] + ' · ' + GROUPS[group]
            content.append('<section id="panel-%d-%s"><h2>%s</h2><p>同一组方法按完整job、餐表、bolus factor与场景逐项对齐。'
                           '粗体仅为本表完整非烟测方法中的数值最优；不代表显著性或模型晋升。</p>' % (i,group,esc(title)))
            if group == 'internal_ablation':
                content.append('<p class="caution">同一算法的部署/内部消融；不计为新增外部算法。</p>')
                content.append('<p>' + '；'.join(esc(labels[p['name']]['label'] + '：' + labels[p['name']]['deployment_variant'])
                                               for p in chosen) + '</p>')
            content.append('<h3>真实 BG 主表</h3>' + score_table(chosen, labels, 'bg', score_csv, cells))
            content.append('<p>所有血糖比例与风险值为患者内病例均值，再汇总患者间 mean ± sample SD（ddof=1）。'
                           '“血糖SD”列中的单轨迹SD沿原评分器ddof=0。±不是训练seed稳定性，也不是95%CI。'
                           'Observed只代表可见前缀，不进入完整方法最优值比较；†事件数可能在未知尾继续增加。'
                           '长严重低糖事件、未知尾及缺失分钟列为输入病例合计。TIR上下界来自未知结局，不能解释为置信区间。</p>')
            extra_rows = []
            for p in chosen:
                extra_rows.append([esc(labels[p['name']]['label'])] + [esc(stat_text(p['observed'][d][k])) for d,k,_ in EXTRA])
                for domain,key,_ in EXTRA:
                    if domain == 'dose':
                        v=p['observed'][domain][key]
                        score_csv.append(dict(split=p['split'],group=group,panel=p['name'],method_id=labels[p['name']]['method_id'],
                            label=labels[p['name']]['label'],signal=domain,metric=key,mean=v['mean'],sample_sd=v['sd'],
                            n_patients=v['n_patients'],complete=p['complete'] is not None,smoke=p['smoke'],
                            failed_episodes=p['failed_episodes'],planned_episodes=p['episodes'],
                            unknown_tail_minutes=p['unknown_tail_intervals']*5,observed_long54_events=None))
            content.append('<h3>均值、输注与动作变化</h3>' + table(['方法']+[x[2] for x in EXTRA],extra_rows) +
                           '<p>Mean BG与剂量不设单向优劣；用量较少不自动代表更好。TV为已交付基础率相邻差绝对值之和，按观测日归一。</p>')
            content.append('<details><summary>传感器 CGM 次表</summary>' + score_table(chosen, labels, 'cgm', score_csv, cells) + '</details></section>')
        content.append('<section><h2>' + esc(SPLITS[first['split']]) + ' · 患者配对差</h2>' + paired_table(bundle, labels, pair_csv))
        gates=[]
        for name,c in bundle['comparisons'].items():
            old=c['original_joint_nonregression'];family=c['metric_family_comparison']
            status_text='通过' if old.get('passed') is True else '未通过' if old.get('applicable') else '不适用'
            family_names={'tir':'TIR','hypoglycemia':'低糖','hyperglycemia':'高糖','variability':'波动'}
            votes={'improved':'改善','worsened':'变差','tied':'相同','tradeoff':'有取舍'}
            family_text='不适用' if family is None else '；'.join(family_names[k]+'：'+votes[v['status']] for k,v in family['families'].items())
            gates.append([esc(labels[name]['label']),esc(labels[c['reference']]['label']),status_text,esc(family_text)])
        content.append(table(['候选','参考','原联合不退步','指标family描述（独立于原joint）'],gates))
        content.append('<p>差值统一为候选减参考。区间来自输入JSON的患者cluster bootstrap，不在构建时重采样。'
                       '未校正多重比较；相关的低糖/高糖指标不视为多项独立成功。失败前缀只给Observed描述差，不给完整病例推断。</p></section>')
    if worlds:
        content.append(world_section(worlds))
    rows=[]
    used={v['method_id'] for v in labels.values()}
    for mid,method in methods.items():
        if mid not in used:
            continue
        info=method['information_support'];action=method['action_support']
        rows.append([esc(method['display_name']),esc(method['algorithm_family']),
                     esc('；'.join(info['input_fields'])),esc(action.get('bounds','未提供')),
                     esc(json.dumps(method.get('budget',{}),ensure_ascii=False)),
                     esc(method.get('reproduction_claim','项目研究候选/适配；不据本登记表认证控制收益'))])
    content.append('<section id="methods"><h2>方法定义与适配边界</h2>' + table(
        ['登记方法','算法家族','可观测输入','动作范围','训练预算登记','复现边界'],rows) +
        '<p>登记表是方法定义快照；当前完成/失败状态以本次评分证据为准。旧外部方法的 frozen+projection 是任务适配，'
        '不是作者原生动作部署。均值/argmax部署变体属于同算法内部消融。相同输入范围不表示预训练、数据、表示和预算完全相同。</p></section>')
    failures=[]
    for p in panels.values():
        for c in p['cases']:
            if not c['complete_followup']:
                failures.append([esc(labels[p['name']]['label']),esc(c['key']),esc(str(c['failure_reason'])),
                                 str(c['unknown_tail_minutes']),fmt(c['metrics']['bg']['coverage_pct']),
                                 '['+fmt(c['metrics']['bg'].get('tir_lower_bound_pct'))+', '+fmt(c['metrics']['bg'].get('tir_upper_bound_pct'))+']'])
    content.append('<section id="failures"><h2>失败与覆盖审计</h2>' + (table(
        ['方法','病例','失败/不完整原因','未知尾分钟','BG覆盖 %','TIR上下界 %'],failures) if failures else
        '<p>当前输入中没有失败或不完整病例。此陈述仅覆盖已输入的结果，不代表其他运行或未知后续。</p>') +
        '<p><a href="病例审计.csv">全部病例、失败、覆盖、未知尾与原始SHA</a> · '
        '<a href="指标长表.csv">BG/CGM/剂量精确数值CSV</a> · <a href="患者配对差.csv">患者配对差CSV</a></p></section>')
    source_rows=[]
    for item in sources:
        source_rows.append('<li><a href="'+esc(item['copy'],quote=True)+'">'+esc(item['role'])+'</a><br><span class="mono">'+
                           esc(item['original'])+'<br>SHA256 '+item['sha256']+'</span></li>')
    content.append('<section id="sources"><h2>来源与构建身份</h2><p>来源JSON随报告离线打包。构建器只检查汇总及绑定，'
                   '不重新运行模型/模拟器/原评分器；原始轨迹没有复制进此轻量报告，逐病例引用与SHA保留在CSV及来源JSON中。</p><ul class="source-list">'+
                   ''.join(source_rows)+'</ul><p><a href="构建清单.json">构建manifest及输出SHA</a> · '
                   '<a href="核查结果.json">机器数据与HTML核查</a></p></section>')
    navigation = '<a href="#reading">当前解读</a>'
    for i, (_, bundle) in enumerate(bundles):
        for group in GROUPS:
            if any(labels[n]['group'] == group for n in bundle['panel_order']):
                navigation += '<a href="#panel-%d-%s">%s</a>' % (i, group, esc(GROUPS[group]))
    if worlds:
        navigation += '<a href="#world">World诊断</a>'
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'+
        '<title>'+esc(state)+' · 低血糖研究</title><style>'+CSS+'</style></head><body><header><div class="eyebrow">DSENET / CONTROL RESEARCH</div>'+
        '<h1>低血糖控制研究报告</h1><span class="status">'+esc(state)+'</span><p class="lead">公开仿真 · 已知虚拟成人 · 单训练seed · 原独立评分器</p>'+
        '<nav>'+navigation+'<a href="#methods">方法边界</a><a href="#failures">失败审计</a><a href="#sources">来源</a></nav></header><main>'+''.join(content)+
        '</main><footer>研究证据不等于真实患者准入或临床安全证明。报告状态不自动改变任何模型或产品发布状态。</footer></body></html>')


def write_csv(path, rows):
    require(rows, 'No CSV rows to export')
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',action='append',default=[],help='Audited nonconfirmation checks/panels_*.json; repeat')
    ap.add_argument('--confirmation',action='append',default=[],help='Explicit audited confirmation checks/panels_*.json')
    ap.add_argument('--world',nargs=2,action='append',default=[],metavar=('DIAGNOSTICS','PREDICTION_BINDING'))
    ap.add_argument('--actions',action='append',default=[],help='Optional checks action diagnostic bound to an included panel')
    ap.add_argument('--registry',default='method_registry.json')
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--summary-file',type=Path)
    group.add_argument('--summary-json',help='Reviewable JSON prose/method mapping; saved separately in output')
    ap.add_argument('--status',choices=['interim','final'],default='interim')
    ap.add_argument('--output',default='delivery',help='New directory relative to research root; never overwrite')
    args=ap.parse_args()
    protocol=read_json(R/'protocol.json')
    require(protocol['scorer_sha256']==SCORER_SHA and sha(P/protocol['scorer'])==SCORER_SHA, 'Frozen scorer bytes changed')
    paths=[(checked_path(x,True),False) for x in args.panel]+[(checked_path(x,True),True) for x in args.confirmation]
    require(paths and len({p for p,_ in paths})==len(paths), 'Need unique audited panel inputs')
    bundles=[(path,load_panels(path,protocol,confirmation)) for path,confirmation in paths]
    panels={}
    for _,data in bundles:
        for n,panel in data['panels'].items():
            require(n not in panels,'Repeated panel across files: '+n);panels[n]=panel
    registry_path=Path(args.registry)
    if not registry_path.is_absolute():registry_path=R/registry_path
    registry_path=registry_path.resolve();registry_path.relative_to(R)
    registry=read_json(registry_path);require(registry['schema_version']==1,'Unsupported registry schema')
    summary=read_json(args.summary_file) if args.summary_file else parse_json(args.summary_json)
    methods=validate_summary(summary,panels,registry,args.status,args.confirmation)
    action_paths=[checked_path(x) for x in args.actions]
    actions=[validate_actions(read_json(path),panels) for path in action_paths]
    worlds=[];world_paths=[]
    for d,b in args.world:
        d,b=checked_path(d),checked_path(b);worlds.append((validate_world(read_json(d),read_json(b),protocol),read_json(b)));world_paths.extend([d,b])
    if len(worlds)>1:
        first=worlds[0][1]
        require(all(w[1]['datasets'][k]['file_order']==first['datasets'][k]['file_order']
                    for w in worlds[1:] for k in ('natural','paired')), 'World comparisons use different validation data')
    output=(R/args.output).resolve();output.relative_to(R)
    require(output!=R and not output.exists(),'Output must be a new research subdirectory')
    output.mkdir(parents=True,exist_ok=False);(output/'sources').mkdir()
    input_paths=[p for p,_ in paths]+world_paths+action_paths+[registry_path,R/'protocol.json',Path(__file__),R/'summarize_panels.py',P/protocol['scorer']]
    if args.summary_file:input_paths.append(args.summary_file.resolve())
    if (R/'report_design.md').is_file():input_paths.append(R/'report_design.md')
    sources=[]
    for i,path in enumerate(dict.fromkeys(input_paths)):
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(path.resolve() not in READ_HASHES or READ_HASHES[path.resolve()]==digest, 'Input changed after validation')
        target=output/'sources'/('%02d_'%i+path.name);target.write_bytes(data)
        sources.append(dict(role=path.name,original=str(path),copy=str(target.relative_to(output)),sha256=digest))
    write_json(output/'报告解读.json',summary)
    score_rows=[];pair_rows=[];cells={}
    document=render(bundles,worlds,summary,methods,args.status,sources,score_rows,pair_rows,cells,actions)
    html_path=output/'阶段研究报告.html'
    with html_path.open('x',encoding='utf-8') as stream:stream.write(document)
    write_csv(output/'指标长表.csv',score_rows)
    if pair_rows:write_csv(output/'患者配对差.csv',pair_rows)
    else:
        with (output/'患者配对差.csv').open('x',encoding='utf-8-sig') as stream:stream.write('candidate,reference,metric\n')
    case_rows=[]
    for n,panel in panels.items():
        for c in panel['cases']:
            case_rows.append(dict(panel=n,split=panel['split'],method_id=summary['panels'][n]['method_id'],
                case=c['key'],patient=c['job']['patient'],seed=c['job']['seed'],bolus_factor=c['job']['bolus_factor'],
                complete=c['complete_followup'],failure_reason=c['failure_reason'],technical_failure_reason=c['technical_failure_reason'],
                observed_intervals=c['observed_record_intervals'],unknown_tail_minutes=c['unknown_tail_minutes'],
                bg_coverage_pct=c['metrics']['bg']['coverage_pct'],cgm_coverage_pct=c['metrics']['cgm']['coverage_pct'],
                bg_tir_lower=c['metrics']['bg'].get('tir_lower_bound_pct'),bg_tir_upper=c['metrics']['bg'].get('tir_upper_bound_pct'),
                raw_path=c['raw_path'],raw_sha256=c['raw_sha256']))
    write_csv(output/'病例审计.csv',case_rows)
    # Check the actual emitted HTML and lossless CSV, not only render inputs.
    from html.parser import HTMLParser
    class AuditHTML(HTMLParser):
        def __init__(self):
            super().__init__();self.active=None;self.text=[];self.found={};self.links=[];self.script=False;self.best=set();self.ids=set()
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='script':self.script=True
            if 'href' in a:self.links.append(a['href'])
            if 'data-cell' in a:self.active=a['data-cell'];self.text=[]
            if tag=='b' and a.get('class')=='best':
                require(self.active is not None,'Best marker outside a verified metric cell');self.best.add(self.active)
            if 'id' in a:
                require(a['id'] not in self.ids,'Duplicate HTML anchor');self.ids.add(a['id'])
        def handle_data(self,data):
            if self.active is not None:self.text.append(data)
        def handle_endtag(self,tag):
            if tag=='span' and self.active is not None:
                require(self.active not in self.found,'Duplicate HTML metric cell')
                self.found[self.active]=''.join(self.text);self.active=None
    parsed=AuditHTML();parsed.feed(html_path.read_text(encoding='utf-8'))
    require(not parsed.script and set(parsed.found)==set(cells),'HTML cells/scripts mismatch')
    require(all(parsed.found[k]==v['text'] for k,v in cells.items()),'HTML numeric transcription mismatch')
    require(parsed.best=={k for k,v in cells.items() if v['best']},'HTML bold markers disagree with complete-only best values')
    require(all(not v['best'] or panels[v['panel']]['complete'] is not None for v in cells.values()),'Failed panel bolded as best')
    require(all(not u.startswith(('http:','https:','//')) for u in parsed.links),'Report is not offline-contained')
    for link in parsed.links:
        if link.startswith('#'):
            require(link[1:] in parsed.ids,'Broken HTML navigation anchor')
        elif link not in ('构建清单.json','核查结果.json'):
            require((output/link).is_file(),'Broken local output link: '+link)
    with (output/'指标长表.csv').open(encoding='utf-8-sig',newline='') as stream:
        exported=list(csv.DictReader(stream))
    require(len(exported)==len(score_rows),'CSV row count mismatch')
    for a,b in zip(exported,score_rows):
        require(all(a[k]==('' if v is None else str(v)) for k,v in b.items()),'CSV lost numeric evidence')
    check=dict(status='passed',standard_library_only=True,raw_scorer_executed=False,model_or_simulator_executed=False,
        audited_panels=len(panels),audited_cases=len(case_rows),html_numeric_cells=len(cells),lossless_csv_rows=len(score_rows),
        html_transcription_exact=True,failed_panels_excluded_from_bolding=True,offline_links_verified=True,
        aggregate_consistency_tolerance=dict(relative=1e-10,absolute=1e-9,raw_score_audit_relaxed=False),
        visual_browser_review='pending_parent',no_model_promotion=True)
    write_json(output/'核查结果.json',check)
    outputs={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'构建清单.json',dict(schema=1,status=args.status,built_utc=datetime.now(timezone.utc).isoformat(),
        builder_sha256=sha(Path(__file__)),scorer_sha256=SCORER_SHA,protocol_sha256=sha(R/'protocol.json'),
        source_inputs=sources,outputs_sha256=outputs,panels={n:dict(split=p['split'],smoke=p['smoke'],
            manifest_sha256=p['manifest_sha256'],summary_sha256=p['summary_sha256'],checkpoint_sha256=p['checkpoint_sha256'])
            for n,p in panels.items()},model_promoted=False,automatic_SOTA_claim=False,
        visual_review='pending_parent',source_raw_trajectories_bundled=False))
    print(json.dumps(dict(check,build_status='built_and_machine_checked',output=str(output),html=str(html_path)),ensure_ascii=False))


if __name__=='__main__':
    main()
