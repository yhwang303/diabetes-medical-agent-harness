"""Offline oracle error decomposition; never a deployable controller result.

Swap the same-origin reference trajectory or action effect with labels to
separate shared reference error from relative action-response error. All four
reconstructions are evaluated on identical complete world-validation groups.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import diagnose_world as diagnostic
from diagnose_world import R, sha, load_labels, load_predictions, error_stats

# The shared diagnostic module deliberately initializes NumPy only in its CLI.
# Initialize that existing dependency for library use without changing its file.
diagnostic.np = np


def reconstructions(predicted, actual):
    return dict(predicted_reference_predicted_response=predicted,
        true_reference_predicted_response=actual[:, :1] + predicted - predicted[:, :1],
        predicted_reference_true_response=predicted[:, :1] + actual - actual[:, :1],
        true_reference_true_response=actual)


def statistics(predicted, cgm, bg):
    pred_min = predicted.min(-1)
    cgm_min = cgm.min(-1)
    bg_min = bg.min(-1)
    events = {}
    for threshold in (70, 54):
        positive = cgm_min < threshold
        forecast = pred_min < threshold
        tp = int((positive & forecast).sum())
        fn = int((positive & ~forecast).sum())
        fp = int((~positive & forecast).sum())
        tn = int((~positive & ~forecast).sum())
        events[str(threshold)] = dict(tp=tp, fn=fn, fp=fp, tn=tn,
            fnr=fn / (tp + fn) if tp + fn else None,
            fpr=fp / (tn + fp) if tn + fp else None)
    low = bg_min < 54
    return dict(cgm_point_error=error_stats(predicted - cgm),
        cgm_minimum_error=error_stats(pred_min - cgm_min),
        bg54_subset_cgm_minimum_error=error_stats((pred_min - cgm_min)[low]),
        bg54_subset_minimum_overestimates=int(((pred_min > cgm_min) & low).sum()),
        cgm_horizon_events=events,
        endpoint_error_by_minutes={str(minutes): error_stats(predicted[..., minutes // 5 - 1] - cgm[..., minutes // 5 - 1])
                                   for minutes in (15, 30, 60, 120, 240, 360)})


def run(variant):
    folder = R / 'results' / ('world_' + variant + '_r1')
    binding_path = R / 'checks' / ('world_' + variant + '_r1_best_prediction_binding.json')
    binding = json.loads(binding_path.read_text())
    provenance_path = folder / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    checkpoint = folder / 'best.pt'
    if (binding['status'] != 'passed' or not binding['passed'] or binding['split'] != 'world_validation'
            or binding['checkpoint_role'] != 'best' or binding['checkpoint'] != str(checkpoint)):
        raise ValueError('Expected the independently verified best world checkpoint')
    protocol_path = R / 'protocol.json'
    for path in (checkpoint, provenance_path, protocol_path):
        if sha(path) != binding['input_and_source_sha256'][str(path)]:
            raise ValueError('Previously verified model/provenance/protocol changed')
    protocol = json.loads(protocol_path.read_text())
    labels_binding = provenance['data']['paired_validation']
    data, identity = load_labels(Path(labels_binding['path']), labels_binding, True, protocol)
    prediction_path = Path(binding['datasets']['paired']['predictions'])
    if sha(prediction_path) != binding['datasets']['paired']['prediction_sha256']:
        raise ValueError('Previously recomputed prediction bytes changed')
    q, _, levels = load_predictions(prediction_path, len(data['mask']), 9, provenance['config']['quantiles'])
    complete = data['mask'].all(axis=(1, 2))
    predicted = q[complete, ..., int(np.argmin(abs(levels - .5)))]
    cgm = data['target_cgm_mg_dl'][complete].astype(np.float64)
    bg = data['target_bg_mg_dl'][complete].astype(np.float64)
    results = {name: statistics(value, cgm, bg) for name, value in reconstructions(predicted, cgm).items()}
    assert results['true_reference_true_response']['cgm_point_error']['mae_mg_dl'] == 0
    patients = np.concatenate([np.full(item['sample_index_stop'] - item['sample_index_start'], item['job']['patient'])
                               for item in identity['file_order']])[complete]
    return dict(schema=1, status='completed_offline_oracle_diagnostic', variant=variant,
        split='world_validation', checkpoint_sha256=sha(checkpoint), step=binding['step'],
        binding_report_sha256=sha(binding_path), prediction_sha256=sha(prediction_path),
        source_sha256=sha(Path(__file__).resolve()), diagnostic_dependency_sha256=sha(R / 'diagnose_world.py'),
        paired_manifest_sha256=identity['manifest_sha256'], reference_arm=0,
        complete_groups=int(complete.sum()), excluded_incomplete_groups=int((~complete).sum()),
        complete_arms=int(complete.sum() * 9), true_bg54_arms=int((bg.min(-1) < 54).sum()),
        true_bg54_arms_by_virtual_patient={str(int(p)): int((bg[patients == p].min(-1) < 54).sum()) for p in np.unique(patients)},
        reconstructions=results,
        limitations=['Only predicted_reference_predicted_response is a real model prediction.',
            'Oracle substitutions use future labels and cannot be deployed, trained on here, or counted as candidate control performance.',
            'CGM threshold event errors compare CGM with CGM; BG54 only defines a diagnostic subgroup.',
            'Reference and response errors interact through nonlinear minima and thresholds; no additive attribution of FNR is asserted.',
            'Overlapping windows and same-origin arms are dependent; no independent-sample inference or clinical claim.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['point', 'quantile'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = (R / args.output).resolve()
    output.relative_to((R / 'checks').resolve())
    # An isolated common bias must cancel only when the reference is replaced.
    actual = np.array([[[100., 80.], [80., 50.]]])
    sanity = reconstructions(actual + 20., actual)
    assert np.array_equal(sanity['true_reference_predicted_response'], actual)
    assert np.array_equal(sanity['predicted_reference_true_response'], actual + 20.)
    result = run(args.variant)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], output=str(output),
        results={name: {'bg54_subset_minimum_mae': values['bg54_subset_cgm_minimum_error']['mae_mg_dl'],
                       'cgm54_fn': values['cgm_horizon_events']['54']['fn']}
                 for name, values in result['reconstructions'].items()})))


if __name__ == '__main__':
    main()
