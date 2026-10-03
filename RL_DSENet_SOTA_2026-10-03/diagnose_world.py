"""Independent saved-prediction diagnostics on world_validation only.

No Torch/model inference, checkpoint selection, probability recalibration, or
training/development/confirmation label loading. Explicit --step is required.
--self-test uses only synthetic in-memory arrays and writes no artifacts.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

R = Path(__file__).resolve().parent
LEVELS = [.05, .1, .25, .5, .75, .9, .95]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def local_path(value):
    path = Path(value).resolve()
    path.relative_to(R)
    return path


def validate_manifest(manifest, paired, protocol):
    """Reject forbidden split/seed metadata before opening any label array."""
    if manifest.get('status') != 'completed' or manifest.get('split') != 'world_validation':
        raise ValueError('Only completed world_validation manifests are accepted')
    if manifest.get('horizon') != 72 or (manifest.get('kind') == 'paired_interventions') != paired:
        raise ValueError('Dataset horizon or natural/paired kind mismatch')
    records = sorted(manifest['episodes'], key=lambda item: item['file'])
    if not records or len({item['file'] for item in records}) != len(records):
        raise ValueError('Empty dataset or duplicate manifest file')
    for item in records:
        name = Path(item['file']); job = item['job']
        if name.name != str(name) or name.suffix != '.npz':
            raise ValueError('Manifest entries must be local NPZ filenames')
        if (job['seed'] not in protocol['world_validation_scenario_seeds']
                or job['patient'] not in protocol['patients']
                or job['bolus_factor'] not in protocol['bolus_factors']):
            raise ValueError('Patient/scenario outside the frozen world_validation protocol')
    return records


def load_labels(directory, binding, paired, protocol):
    directory = local_path(directory); manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = validate_manifest(manifest, paired, protocol)
    if (binding['split'] != 'world_validation' or binding['paired'] != paired
            or sha(manifest_path) != binding['manifest_sha256']):
        raise ValueError('Dataset manifest differs from the training provenance')
    order = [dict(file=item['file'], count=item['groups' if paired else 'origins'],
                  sha256=item['sha256'], job=item['job']) for item in records]
    if order != binding['file_order']:
        raise ValueError('Saved prediction sample order is not bound to this dataset')
    chunks = {key: [] for key in ('target_cgm_mg_dl', 'target_bg_mg_dl', 'mask')}
    index = []; count = 0
    for item in order:
        path = (directory / item['file']).resolve(); path.relative_to(directory)
        if sha(path) != item['sha256']: raise ValueError('Label NPZ hash mismatch: ' + item['file'])
        with np.load(path, allow_pickle=False) as packed:
            # No histories, actions, future meal/bolus, or other label-only arrays are read.
            arrays = {key: packed[key] for key in chunks}
            origins = packed['origin_minute']
        n = item['count']; expected = (n, 9, 72) if paired else (n, 72)
        if n < 1 or origins.shape != (n,) or not np.isfinite(origins).all():
            raise ValueError('Origin identity/count mismatch')
        for key, value in arrays.items():
            if value.shape != expected or value.dtype != (np.bool_ if key == 'mask' else np.float32):
                raise ValueError('Unexpected label shape/dtype: ' + key)
            if key != 'mask' and not np.isfinite(value).all(): raise ValueError('Nonfinite stored label')
            chunks[key].append(value if paired else value[:, None])
        mask = arrays['mask']
        if not mask.any(-1).all() or (mask[..., 1:] & ~mask[..., :-1]).any():
            raise ValueError('Labels must have nonempty observed prefixes and unknown padded tails')
        index.append(dict(file=item['file'], sha256=item['sha256'], job=item['job'],
                          sample_index_start=count, sample_index_stop=count + n,
                          origins=n, first_origin_minute=float(origins[0]), last_origin_minute=float(origins[-1])))
        count += n
    data = {key: np.concatenate(values) for key, values in chunks.items()}
    if count != binding['groups'] or data['mask'].shape[1] != binding['arms']:
        raise ValueError('Training provenance group/arm count mismatch')
    return data, dict(directory=str(directory), original_directory=binding['path'],
                      manifest_sha256=sha(manifest_path), file_order=index,
                      patients=sorted({item['job']['patient'] for item in order}),
                      scenarios=sorted({item['job']['seed'] for item in order}),
                      episodes=len(order))


def load_predictions(path, groups, arms, expected_levels):
    with np.load(path, allow_pickle=False) as packed:
        q = packed['cgm_quantiles_mgdl']; p = packed['bg_event_probabilities']
        levels = packed['quantile_levels']; indices = packed['sample_index']
    if (q.shape != (groups, arms, 72, len(expected_levels)) or p.shape != (groups, arms, 3)
            or levels.shape != (len(expected_levels),) or not np.allclose(levels, expected_levels, rtol=0, atol=1e-7)
            or not np.array_equal(indices, np.arange(groups))):
        raise ValueError('Prediction shape, quantile levels, or ordered sample indices mismatch')
    if (not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.allclose(p.sum(-1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid/crossing quantiles or categorical BG event probabilities')
    return q.astype(np.float64), p.astype(np.float64), levels.astype(np.float64)


def mean(values):
    return float(np.mean(values)) if np.size(values) else None


def fraction(count, total):
    return float(count / total) if total else None


def error_stats(error):
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    return dict(count=len(error), bias_mg_dl=mean(error), mae_mg_dl=mean(np.abs(error)),
                rmse_mg_dl=float(np.sqrt(np.mean(error ** 2))) if len(error) else None)


def event_metrics(probability, target, complete, threshold):
    """Full-horizon event metrics exclude all censored windows, including positives."""
    actual = (target[complete] < threshold).astype(np.float64)
    p = probability[complete]
    positive = int(actual.sum()); negative = len(actual) - positive
    fn = int(((actual == 1) & (p < .5)).sum()); fp = int(((actual == 0) & (p >= .5)).sum())
    bins = []
    for k in range(10):
        take = (p >= k / 10) & ((p < (k + 1) / 10) if k < 9 else (p <= 1))
        bins.append(dict(lower=k / 10, upper=(k + 1) / 10, upper_inclusive=k == 9,
                         count=int(take.sum()), predicted_mean=mean(p[take]), observed_rate=mean(actual[take])))
    return dict(complete_windows=len(actual), positive=positive, negative=negative,
                brier=mean((p - actual) ** 2), predicted_mean=mean(p), observed_rate=fraction(positive, len(actual)),
                threshold=.5, false_negatives=fn, false_positives=fp,
                true_positives=positive - fn, true_negatives=negative - fp,
                fnr=fraction(fn, positive), fpr=fraction(fp, negative), calibration_bins=bins,
                censored_windows_excluded=int((~complete).sum()),
                censored_observed_positive=int(((target < threshold) & ~complete).sum()),
                censored_unknown_no_observed_event=int(((target >= threshold) & ~complete).sum()))


def ranking(prediction, target, tolerance=1e-6):
    i, j = np.triu_indices(prediction.shape[1], 1)
    truth = target[:, i] - target[:, j]; predicted = prediction[:, i] - prediction[:, j]
    eligible = np.abs(truth) > tolerance
    correct = int(((np.sign(truth) == np.sign(predicted)) & eligible).sum())
    total = int(eligible.sum())
    return dict(correct_pairs=correct, eligible_pairs=total, accuracy=fraction(correct, total),
                target_ties_excluded=int((~eligible).sum()),
                predicted_exact_ties_as_incorrect=int(((predicted == 0) & eligible).sum()),
                target_tie_tolerance=tolerance)


def dataset_metrics(q, probability, levels, data, paired):
    mask = data['mask']; cgm = data['target_cgm_mg_dl'].astype(np.float64)
    bg = data['target_bg_mg_dl'].astype(np.float64)
    complete = mask.all(-1); n, arms, horizon = mask.shape
    median_index = int(np.flatnonzero(np.isclose(levels, .5, rtol=0, atol=1e-7))[0])
    median = q[..., median_index]; observed_cgm_min = np.where(mask, cgm, np.inf).min(-1)
    observed_bg_min = np.where(mask, bg, np.inf).min(-1)
    p70_raw = probability[..., 1:].sum(-1)
    p70 = np.clip(p70_raw, 0, 1); p54 = probability[..., 2]
    result = dict(
        coverage=dict(groups=n, arms_per_group=arms, arm_windows=n * arms,
                      complete_arm_windows=int(complete.sum()), censored_arm_windows=int((~complete).sum()),
                      complete_groups=int(complete.all(1).sum()), groups_with_censoring=int((~complete.all(1)).sum()),
                      observed_points=int(mask.sum()), planned_points=int(mask.size),
                      unknown_tail_points=int((~mask).sum()), observed_point_fraction=float(mask.mean()),
                      observed_steps_per_arm=dict(minimum=int(mask.sum(-1).min()), maximum=int(mask.sum(-1).max()),
                                                  mean=float(mask.sum(-1).mean()))),
        cgm_median=error_stats((median - cgm)[mask]),
        cgm_observed_low_points={str(t): error_stats((median - cgm)[mask & (cgm < t)]) for t in (70, 54)},
        cgm_quantiles={}, central_pointwise_intervals={}, bg_horizon_events={}, minimum_cgm={})
    for k, level in enumerate(levels):
        error = q[..., k] - cgm; residual = -error
        result['cgm_quantiles'][format(float(level), '.3g')] = dict(
            level=float(level), **error_stats(error[mask]),
            pinball_mg_dl=mean(np.maximum(level * residual, (level - 1) * residual)[mask]),
            empirical_cdf_coverage=mean((cgm <= q[..., k])[mask]))
    for lo, hi in ((.05, .95), (.1, .9), (.25, .75)):
        lower = np.flatnonzero(np.isclose(levels, lo, rtol=0, atol=1e-7))
        upper = np.flatnonzero(np.isclose(levels, hi, rtol=0, atol=1e-7))
        if not len(lower) or not len(upper): continue
        left, right = q[..., lower[0]], q[..., upper[0]]
        result['central_pointwise_intervals'][format(hi - lo, '.2g')] = dict(
            lower_level=lo, upper_level=hi, nominal_pointwise_coverage=hi - lo,
            observed_points=int(mask.sum()), coverage=mean(((cgm >= left) & (cgm <= right))[mask]),
            mean_width_mg_dl=mean((right - left)[mask]),
            median_width_mg_dl=float(np.median((right - left)[mask])))
    for threshold, predicted in ((70, p70), (54, p54)):
        result['bg_horizon_events'][str(threshold)] = event_metrics(predicted, observed_bg_min, complete, threshold)
    result['probability_roundoff'] = dict(maximum_p70_clip_adjustment=float(np.max(np.abs(p70 - p70_raw))),
        maximum_class_sum_error=float(np.max(np.abs(probability.sum(-1) - 1))), fitted_calibration=False)
    # Prefix minimum uses the same observed support on both sides. It is NOT a six-hour minimum.
    minimum_error = np.where(mask, median, np.inf).min(-1) - observed_cgm_min
    for scope, subset in (('complete_horizon', complete), ('censored_observed_prefix_only', ~complete)):
        conditions = dict(all=subset)
        for threshold in (70, 54):
            conditions['bg_observed_below' + str(threshold)] = subset & (observed_bg_min < threshold)
            conditions['cgm_observed_below' + str(threshold)] = subset & (observed_cgm_min < threshold)
        result['minimum_cgm'][scope] = {}
        for name, take in conditions.items():
            stats = error_stats(minimum_error[take])
            stats['minimum_overestimation_fraction'] = mean(minimum_error[take] > 0)
            stats['true_cgm_minimum_mean_mg_dl'] = mean(observed_cgm_min[take])
            stats['marginal_quantile_lower_envelope_bias'] = {
                format(float(level), '.3g'): mean((np.where(mask, q[..., k], np.inf).min(-1) - observed_cgm_min)[take])
                for k, level in enumerate(levels) if level < .5}
            result['minimum_cgm'][scope][name] = stats
    if paired:
        common = mask[:, 1:] & mask[:, :1]
        delta = median[:, 1:] - median[:, :1]
        truth = cgm[:, 1:] - cgm[:, :1]; error = delta - truth
        arm_mae = (np.abs(error) * common).sum(-1) / common.sum(-1)
        full_pairs = complete[:, 1:] & complete[:, :1]
        complete_groups = complete.all(1)
        median_min = median.min(-1); true_min = cgm.min(-1)
        min_delta_error = (median_min[:, 1:] - median_min[:, :1]) - (true_min[:, 1:] - true_min[:, :1])
        result['paired_response'] = dict(reference_arm=0, response_unit='CGM mg/dL relative to same-origin hold arm',
            common_prefix_points=int(common.sum()), planned_pair_points=int(common.size),
            common_prefix_fraction=float(common.mean()), complete_arm_reference_pairs=int(full_pairs.sum()),
            censored_arm_reference_pairs=int((~full_pairs).sum()),
            observed_delta_error=error_stats(error[common]),
            group_macro_mean_arm_delta_mae_mg_dl=float(arm_mae.mean(1).mean()),
            complete_6h_endpoint_delta_error=error_stats(error[..., -1][full_pairs]),
            complete_minimum_delta_error=error_stats(min_delta_error[full_pairs]),
            ranking_complete_groups=int(complete_groups.sum()),
            ranking_censored_groups_excluded=int((~complete_groups).sum()),
            minimum_cgm_order=ranking(median_min[complete_groups], true_min[complete_groups]),
            endpoint_cgm_order=ranking(median[..., -1][complete_groups], cgm[..., -1][complete_groups]),
            bg_event_order={str(t): ranking(p[complete_groups], (observed_bg_min[complete_groups] < t).astype(float))
                            for t, p in ((70, p70), (54, p54))})
    return result


def compare_saved_metrics(result, saved, paired):
    pairs = [('groups', result['coverage']['groups'], saved['groups']),
             ('arms', result['coverage']['arms_per_group'], saved['arms']),
             ('valid_cgm_points', result['coverage']['observed_points'], saved['valid_cgm_points']),
             ('complete_event_windows', result['coverage']['complete_arm_windows'], saved['complete_event_windows']),
             ('cgm_mae', result['cgm_median']['mae_mg_dl'], saved['cgm_mae_mg_dl']),
             ('cgm_rmse', result['cgm_median']['rmse_mg_dl'], saved['cgm_rmse_mg_dl'])]
    if paired:
        pairs.append(('paired_delta_mae', result['paired_response']['group_macro_mean_arm_delta_mae_mg_dl'],
                      saved['paired_delta_mae_mg_dl']))
        pairs.append(('minimum_cgm_order_accuracy', result['paired_response']['minimum_cgm_order']['accuracy'],
                      saved['minimum_cgm_order_accuracy']['accuracy']))
    else:
        for threshold in ('70', '54'):
            pairs.append(('bg' + threshold + '_brier', result['bg_horizon_events'][threshold]['brier'],
                          saved['bg_horizon_event_calibration'][threshold]['brier']))
    checks = []
    for name, calculated, reference in pairs:
        passed = calculated is None and reference is None if calculated is None or reference is None else bool(
            np.isclose(calculated, reference, rtol=1e-5, atol=1e-4))
        checks.append(dict(metric=name, independent=calculated, training_report=reference, passed=passed))
    if not all(item['passed'] for item in checks):
        raise ValueError('Saved prediction/label metrics disagree with trainer report: ' + json.dumps(checks))
    return checks


def diagnose(run, step, natural_directory, paired_directory):
    run = local_path(run)
    protocol_path = R / 'protocol.json'; protocol = json.loads(protocol_path.read_text())
    provenance_path = run / 'provenance.json'; provenance = json.loads(provenance_path.read_text())
    completion_path = run / 'completion.json'; completion = json.loads(completion_path.read_text())
    config = provenance['config']
    if (provenance['seed'] != 260915 or provenance['protocol_sha256'] != sha(protocol_path)
            or completion['status'] != 'training_completed_candidate_only'
            or completion['steps'] != config['steps'] or completion['configured_steps'] != config['configured_steps']
            or completion['budget_override'] != config['budget_override'] or completion['selected_on'] != 'world_validation'):
        raise ValueError('Expected a completed, protocol-bound world training run')
    if type(step) is not int or not 0 < step <= completion['steps']:
        raise ValueError('Explicit saved validation step must fall within the completed training run')
    levels = [.5] if config['variant'] == 'point' else LEVELS
    if config['variant'] not in ('point', 'quantile') or config['quantiles'] != levels:
        raise ValueError('Unknown training quantile variant')
    validation_path = run / 'validation' / ('step%07d.json' % step)
    validation = json.loads(validation_path.read_text())
    if validation['step'] != step or validation['split'] != 'world_validation':
        raise ValueError('Saved validation step/split mismatch')
    results = {}; inputs = {}; checks = {}
    for kind, directory in (('natural', natural_directory), ('paired', paired_directory)):
        paired = kind == 'paired'; binding = provenance['data'][kind + '_validation']
        data, identity = load_labels(directory, binding, paired, protocol)
        prediction_path = run / 'validation' / ('%s_step%07d.npz' % (kind, step))
        q, p, actual_levels = load_predictions(prediction_path, len(data['mask']), data['mask'].shape[1], levels)
        results[kind] = dataset_metrics(q, p, actual_levels, data, paired)
        checks[kind] = compare_saved_metrics(results[kind], validation[kind], paired)
        inputs[kind] = dict(prediction_path=str(prediction_path), prediction_sha256=sha(prediction_path), labels=identity)
    return dict(schema=1, evidence_kind='real_saved_world_validation_predictions', run=str(run), step=step,
        variant=config['variant'], quantile_levels=levels, training_seed=260915,
        training_budget=dict(completed_steps=completion['steps'], configured_steps=completion['configured_steps'],
                             smoke_budget_override=completion['budget_override']),
        inputs=inputs, provenance_sha256=sha(provenance_path), completion_sha256=sha(completion_path),
        saved_validation_json_sha256=sha(validation_path), protocol_sha256=sha(protocol_path),
        diagnostic_source_sha256=sha(Path(__file__).resolve()), numpy_version=np.__version__,
        trainer_metric_reproduction=checks, metrics=results,
        limitations=[
            'CGM trajectory targets are sensor observations; BG event labels are simulator true-glucose threshold crossings.',
            'Event Brier/calibration/FNR use complete 72-step arm windows only; unknown censored tails are never negatives.',
            'Natural and intervention-arm event populations are separate; paired calibration is not natural prevalence calibration.',
            'Overlapping windows, repeated scenarios, same-patient episodes and paired arms are dependent; no iid confidence intervals.',
            'Quantiles and intervals are marginal pointwise predictions, not joint trajectory coverage, minimum quantiles or CVaR.',
            'Positive minimum bias means predicted CGM minimum is higher than the observed minimum; CGM and BG lows differ.',
            'Validation NPZ has no embedded checkpoint SHA; recorded file hashes and trainer-metric agreement bind observed artifacts, not cryptographic prediction-to-weight identity.',
            'No calibration fit, checkpoint/threshold selection, new inference, development/confirmation data or closed-loop benefit test was performed.'])


def self_test():
    """Deterministic mechanism fixtures, not a measured model or patient result."""
    mask = np.ones((3, 1, 72), dtype=bool); mask[2, :, 2:] = False
    cgm = np.full((3, 1, 72), 100., dtype=np.float32); cgm[0, 0, 10] = 50.; cgm[2, 0, 0] = 40.
    bg = np.full_like(cgm, 120.); bg[0, 0, 10] = 60.; bg[2, 0, 0] = 40.
    data = dict(mask=mask, target_cgm_mg_dl=cgm, target_bg_mg_dl=bg)
    q = cgm[..., None].astype(float) + np.array([-5., 0., 5.])
    probability = np.array([[[.6, .3, .1]], [[.2, .6, .2]], [[.9, .05, .05]]])
    result = dataset_metrics(q, probability, np.array([.05, .5, .95]), data, False)
    assert result['cgm_median']['mae_mg_dl'] == 0
    assert result['coverage']['unknown_tail_points'] == 70
    assert result['central_pointwise_intervals']['0.9']['coverage'] == 1
    assert result['central_pointwise_intervals']['0.9']['mean_width_mg_dl'] == 10
    event70 = result['bg_horizon_events']['70']; event54 = result['bg_horizon_events']['54']
    assert event70['positive'] == 1 and event70['negative'] == 1 and event70['false_negatives'] == 1
    assert event70['false_positives'] == 1 and np.isclose(event70['brier'], .5)
    assert event54['positive'] == 0 and event54['fnr'] is None
    assert event54['censored_observed_positive'] == 1 and event54['censored_windows_excluded'] == 1
    assert sum(b['count'] for b in event70['calibration_bins']) == 2
    assert result['minimum_cgm']['complete_horizon']['bg_observed_below54']['count'] == 0
    assert result['minimum_cgm']['complete_horizon']['cgm_observed_below54']['count'] == 1
    # Padded labels/predictions must not change observed-prefix or CGM point metrics.
    q2 = q.copy(); q2[2, :, 2:] = -10000
    altered = dataset_metrics(q2, probability, np.array([.05, .5, .95]), data, False)
    assert altered['cgm_median'] == result['cgm_median'] and altered['minimum_cgm'] == result['minimum_cgm']
    point = dataset_metrics(q[..., 1:2], probability, np.array([.5]), data, False)
    assert point['central_pointwise_intervals'] == {} and point['cgm_median']['mae_mg_dl'] == 0
    # Same-origin paired differences remove a shared 7 mg/dL bias.
    paired_cgm = np.broadcast_to((100 + np.arange(9) * 3)[None, :, None], (2, 9, 72)).copy().astype(np.float32)
    paired_mask = np.ones_like(paired_cgm, dtype=bool); paired_mask[1, 8, 20:] = False
    paired_data = dict(mask=paired_mask, target_cgm_mg_dl=paired_cgm, target_bg_mg_dl=paired_cgm)
    paired_prob = np.broadcast_to([1., 0., 0.], (2, 9, 3)).copy()
    paired_result = dataset_metrics(paired_cgm[..., None] + 7, paired_prob, np.array([.5]), paired_data, True)
    response = paired_result['paired_response']
    assert response['observed_delta_error']['mae_mg_dl'] == 0
    assert response['group_macro_mean_arm_delta_mae_mg_dl'] == 0
    assert response['ranking_complete_groups'] == 1 and response['minimum_cgm_order']['eligible_pairs'] == 36
    assert response['minimum_cgm_order']['accuracy'] == 1
    assert response['bg_event_order']['70']['accuracy'] is None
    assert response['complete_arm_reference_pairs'] == 15
    reversed_rank = ranking(-np.arange(9)[None], np.arange(9)[None])
    assert reversed_rank['correct_pairs'] == 0 and reversed_rank['eligible_pairs'] == 36
    tied = ranking(np.zeros((1, 9)), np.arange(9)[None])
    assert tied['predicted_exact_ties_as_incorrect'] == 36
    empty = event_metrics(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1), dtype=bool), 70)
    assert empty['brier'] is None and empty['fnr'] is None
    edge = event_metrics(np.array([[0., .5, 1.]]), np.array([[100., 60., 60.]]), np.ones((1, 3), dtype=bool), 70)
    assert edge['false_negatives'] == 0 and sum(b['count'] for b in edge['calibration_bins']) == 3
    protocol = json.loads((R / 'protocol.json').read_text())
    manifest = dict(status='completed', split='world_validation', horizon=72,
                    episodes=[dict(file='sample.npz', job=dict(patient=1, seed=103011, bolus_factor=1.))])
    assert len(validate_manifest(manifest, False, protocol)) == 1
    rejected = 0
    for field, value in (('split', 'train'), ('split', 'development'), ('split', 'confirmation'), ('status', 'running')):
        bad = dict(manifest); bad[field] = value
        try: validate_manifest(bad, False, protocol)
        except ValueError: rejected += 1
    for seed in (103001, 103101, 103901):
        bad = dict(manifest, episodes=[dict(file='sample.npz', job=dict(patient=1, seed=seed, bolus_factor=1.))])
        try: validate_manifest(bad, False, protocol)
        except ValueError: rejected += 1
    assert rejected == 7
    # Exercise the actual compressed-prediction reader entirely in RAM.
    q_saved = np.broadcast_to(np.arange(7, dtype=np.float32), (2, 1, 72, 7)).copy()
    p_saved = np.broadcast_to(np.array([.5, .25, .25], dtype=np.float32), (2, 1, 3)).copy()
    payload = dict(cgm_quantiles_mgdl=q_saved, bg_event_probabilities=p_saved,
                   quantile_levels=np.array(LEVELS, dtype=np.float32), sample_index=np.arange(2))
    def packed(values):
        buffer = io.BytesIO(); np.savez_compressed(buffer, **values); buffer.seek(0)
        return buffer
    loaded, _, _ = load_predictions(packed(payload), 2, 1, LEVELS)
    assert loaded.shape == (2, 1, 72, 7)
    rejected_predictions = 0
    for key, value in (('sample_index', np.array([1, 0])),
                       ('cgm_quantiles_mgdl', q_saved[..., ::-1]),
                       ('bg_event_probabilities', p_saved * 2),
                       ('quantile_levels', np.array(LEVELS) + .01)):
        bad = dict(payload); bad[key] = value
        try: load_predictions(packed(bad), 2, 1, LEVELS)
        except ValueError: rejected_predictions += 1
    assert rejected_predictions == 4
    # The fixture output itself must remain standards-compliant JSON, with no NaN.
    json.dumps(dict(natural=result, paired=paired_result, empty=empty), allow_nan=False)
    return dict(status='passed', evidence_kind='synthetic_mechanism_tests_only',
                checked=['CGM/BG label distinction', 'complete-only event denominators', 'zero-positive FNR is null',
                         'censored positives never become negatives', 'padding ignored', 'point interval unavailable',
                         'paired reference cancellation', 'complete-group ranking', 'ties/empty denominators',
                         'probability 0/.5/1 bin boundaries', 'seven forbidden metadata cases',
                         'compressed prediction reader and four corrupt-prediction cases', 'finite JSON'],
                real_saved_predictions_loaded=False, numpy_version=np.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run'); parser.add_argument('--step', type=int)
    parser.add_argument('--natural-data'); parser.add_argument('--paired-data'); parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    global np
    import numpy as np
    if args.self_test:
        if any((args.run, args.step, args.natural_data, args.paired_data, args.output)):
            parser.error('--self-test must be used alone; it writes no files')
        print(json.dumps(self_test(), indent=2, allow_nan=False)); return
    if any(value is None for value in (args.run, args.step, args.natural_data, args.paired_data, args.output)):
        parser.error('--run --step --natural-data --paired-data --output are all required')
    output = local_path(args.output)
    if output.exists(): raise FileExistsError(str(output))
    result = diagnose(args.run, args.step, args.natural_data, args.paired_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='completed_diagnostics_only', output=str(output), sha256=sha(output),
                          step=args.step, variant=result['variant'], evidence_kind=result['evidence_kind'])))


if __name__ == '__main__':
    main()
