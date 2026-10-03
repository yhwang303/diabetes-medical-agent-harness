r"""Bind ALL saved world_validation predictions to a specified completed world.

Run with the original CUDA/Mamba Python 3.8 runtime, from the project root:
  .venv-native/bin/python RL_DSENet_SOTA_2026-10-03/verify_world_predictions.py \
    --checkpoint /absolute/research/results/world_quantile_r1/best.pt \
    --output /absolute/research/checks/world_quantile_r1_best_prediction_binding.json

Only explicit best.pt or last.pt is accepted; its embedded step determines the
saved prediction files. No checkpoint is selected, no tolerance is tuned, and
no original artifact is changed. Completed smoke budgets remain marked smoke.
Every CGM quantile and every one of the three BG class probabilities is checked,
including predictions in censored target tails. Numerical agreement is required
at fixed absolute tolerances (CGM 1e-3 mg/dL, probability 1e-6, relative zero);
byte identity is not required or assumed. A failed comparison still writes a
new failure report and exits nonzero. Structural/runtime failures preserve the
partial audit with status=error. Existing report files are never overwritten.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

R = Path(__file__).resolve().parent
P = R.parent
CGM_ATOL = 1e-3
PROBABILITY_ATOL = 1e-6
INPUT_KEYS = ('history', 'anchor_u_h', 'actions_u_h')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def inspect_dataset(binding, paired, protocol):
    """Reject nonvalidation metadata before opening any NPZ array."""
    directory = Path(binding['path']).resolve()
    directory.relative_to((R / 'cache').resolve())
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (binding['split'] != 'world_validation' or binding['paired'] != paired
            or manifest.get('split') != 'world_validation' or manifest.get('status') != 'completed'
            or manifest.get('horizon') != 72
            or (manifest.get('kind') == 'paired_interventions') != paired
            or sha(manifest_path) != binding['manifest_sha256']):
        raise ValueError('Expected the completed, bound world_validation dataset')
    records = sorted(manifest['episodes'], key=lambda item: item['file'])
    if not records or len({item['file'] for item in records}) != len(records):
        raise ValueError('Empty dataset or duplicate manifest file')
    order = []
    for record in records:
        name = Path(record['file']); job = record['job']
        if name.name != str(name) or name.suffix != '.npz':
            raise ValueError('Dataset entries must be local NPZ filenames')
        (directory / name).resolve().relative_to(directory)
        if (job['seed'] not in protocol['world_validation_scenario_seeds']
                or job['patient'] not in protocol['patients']
                or job['bolus_factor'] not in protocol['bolus_factors']):
            raise ValueError('Dataset scenario/patient is outside world_validation')
        count = record['groups' if paired else 'origins']
        if type(count) is not int or count < 1: raise ValueError('Invalid origin count')
        order.append(dict(file=record['file'], count=count, sha256=record['sha256'], job=job))
    if order != binding['file_order'] or sum(item['count'] for item in order) != binding['groups']:
        raise ValueError('Dataset file order/count differs from the saved-prediction provenance')
    if binding['arms'] != (9 if paired else 1): raise ValueError('Dataset arm count mismatch')
    return directory, manifest_path, order


def load_inputs(inspection, paired):
    """Load only allowed inference inputs; retain the original global row order."""
    directory, _, order = inspection
    chunks = {key: [] for key in INPUT_KEYS}
    for item in order:
        path = directory / item['file']
        if sha(path) != item['sha256']: raise ValueError('Dataset NPZ hash mismatch: ' + str(path))
        with np.load(path, allow_pickle=False) as packed:
            arrays = {key: packed[key] for key in INPUT_KEYS}
        count = item['count']
        shapes = dict(history=(count, 72, 22), anchor_u_h=(count,),
                      actions_u_h=(count, 9, 72) if paired else (count, 72))
        for key, value in arrays.items():
            if value.dtype != np.float32 or value.shape != shapes[key] or not np.isfinite(value).all():
                raise ValueError('Input shape/dtype/finiteness mismatch: ' + key)
        if (arrays['anchor_u_h'] < 0).any() or (arrays['actions_u_h'] < 0).any():
            raise ValueError('Negative observed anchor or future basal action')
        for key, value in arrays.items():
            chunks[key].append(value[:, None] if key == 'actions_u_h' and not paired else value)
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def array_hasher(shape, dtype):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(shape=list(shape), dtype=str(dtype)), sort_keys=True).encode('utf-8'))
    digest.update(b'\n')
    return digest


def array_sha(value):
    digest = array_hasher(value.shape, value.dtype)
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        raw = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8')); digest.update(array_sha(raw).encode('ascii'))
    return digest.hexdigest()


class Comparison:
    def __init__(self, reference, tolerance):
        self.reference = reference
        self.tolerance = tolerance
        self.count = 0; self.exceeding = 0; self.nonfinite = 0
        self.maximum = 0.; self.maximum_location = None; self.total_error = 0.
        self.digest = array_hasher(reference.shape, reference.dtype)

    def update(self, begin, computed):
        reference = self.reference[begin:begin + len(computed)]
        if computed.shape != reference.shape or computed.dtype != reference.dtype:
            raise ValueError('Recomputed prediction shape/dtype mismatch')
        self.digest.update(np.ascontiguousarray(computed).tobytes())
        # FP64 subtraction avoids rounding the comparison itself to FP32.
        finite = np.isfinite(reference) & np.isfinite(computed)
        difference = np.zeros(reference.shape, dtype=np.float64)
        np.subtract(computed, reference, out=difference, where=finite, dtype=np.float64)
        difference = np.abs(difference)
        self.nonfinite += int((~finite).sum())
        self.exceeding += int(((difference > self.tolerance) & finite).sum())
        self.count += int(reference.size); self.total_error += float(difference[finite].sum())
        if finite.any():
            index = np.unravel_index(int(np.where(finite, difference, -1.).argmax()), difference.shape)
            error = float(difference[index])
            if self.maximum_location is None or error > self.maximum:
                self.maximum = error
                self.maximum_location = [int(index[0]) + begin] + [int(x) for x in index[1:]]

    def result(self):
        full = self.count == self.reference.size
        finite_count = self.count - self.nonfinite
        return dict(passed=bool(full and self.exceeding == 0 and self.nonfinite == 0),
                    compared_elements=self.count, expected_elements=int(self.reference.size),
                    all_elements_checked=bool(full), absolute_tolerance=self.tolerance, relative_tolerance=0.,
                    elements_exceeding_tolerance=self.exceeding, nonfinite_element_pairs=self.nonfinite,
                    maximum_absolute_error=self.maximum if finite_count else None,
                    maximum_error_index=self.maximum_location,
                    mean_absolute_error=self.total_error / finite_count if finite_count else None,
                    saved_array_sha256=array_sha(self.reference), recomputed_array_sha256=self.digest.hexdigest())


def verify_dataset(model, prediction_path, inspection, paired, batch_size, levels):
    data = load_inputs(inspection, paired)
    groups = len(data['history']); arms = 9 if paired else 1
    with np.load(prediction_path, allow_pickle=False) as packed:
        q = packed['cgm_quantiles_mgdl']; probability = packed['bg_event_probabilities']
        saved_levels = packed['quantile_levels']; indices = packed['sample_index']
    if (q.shape != (groups, arms, 72, len(levels)) or probability.shape != (groups, arms, 3)
            or q.dtype != np.float32 or probability.dtype != np.float32
            or saved_levels.dtype != levels.dtype or not np.array_equal(saved_levels, levels)
            or indices.dtype.kind not in 'iu' or not np.array_equal(indices, np.arange(groups))):
        raise ValueError('Saved prediction shape/dtype/levels/sample order mismatch')
    q_compare = Comparison(q, CGM_ATOL); p_compare = Comparison(probability, PROBABILITY_ATOL)
    batches = 0; started = time.time()
    # This exactly retains trainer batching over concatenated files, including the last short batch.
    with torch.no_grad():
        for begin in range(0, groups, batch_size):
            end = min(begin + batch_size, groups)
            batch = {key: torch.as_tensor(value[begin:end], device='cuda') for key, value in data.items()}
            output = model(batch['history'], batch['anchor_u_h'], batch['actions_u_h'])
            if not np.array_equal(output['quantile_levels'].cpu().numpy(), saved_levels):
                raise ValueError('Model quantile order changed during verification')
            q_compare.update(begin, output['cgm_quantiles_mgdl'].cpu().numpy())
            p_compare.update(begin, output['bg_event_probabilities'].cpu().numpy())
            batches += 1
            if batches == 1 or batches % 50 == 0 or end == groups:
                print(json.dumps(dict(event='verification_progress', kind='paired' if paired else 'natural',
                    checked_groups=end, total_groups=groups, batches=batches, seconds=time.time() - started)), flush=True)
    q_result, p_result = q_compare.result(), p_compare.result()
    return dict(passed=q_result['passed'] and p_result['passed'], groups=groups, arms=arms, horizon=72,
                original_validation_batch_groups=batch_size, recomputed_batches=batches,
                final_batch_groups=groups % batch_size or min(groups, batch_size),
                sample_order_exact=True, quantile_levels_exact=True,
                sample_index_sha256=array_sha(indices), quantile_levels_sha256=array_sha(saved_levels),
                cgm_quantiles=q_result, bg_three_class_probabilities=p_result,
                class_order=['no_BG_below70', 'BG_below70_but_no_BG_below54', 'any_BG_below54'],
                compared_unknown_target_tail_predictions=True, seconds=time.time() - started)


def run_verification(checkpoint, report):
    from world_control_worker import load_world, inspect_provenance
    checkpoint = checkpoint.resolve()
    checkpoint.relative_to((R / 'results').resolve())
    if checkpoint.name not in ('best.pt', 'last.pt'):
        raise ValueError('Specify an existing best.pt or last.pt explicitly; no checkpoint selection is performed')
    provenance_path, provenance, _, artifacts = inspect_provenance(checkpoint)
    completion_path = provenance_path.parent / 'completion.json'
    completion = json.loads(completion_path.read_text())
    role = checkpoint.stem
    if sha(checkpoint) != completion[role + '_checkpoint_sha256']:
        raise ValueError('Explicit checkpoint filename does not match its completion SHA binding')
    config = provenance['config']; protocol = json.loads((R / 'protocol.json').read_text())
    batch_size = config['validation_batch_groups']
    if type(batch_size) is not int or batch_size < 1: raise ValueError('Invalid original validation batch size')
    inspections = {kind: inspect_dataset(provenance['data'][kind + '_validation'], kind == 'paired', protocol)
                   for kind in ('natural', 'paired')}
    model, summary = load_world(checkpoint)  # Absolute path, strict source/P03/norm/frozen-tensor checks.
    step = summary['training_step']
    expected_step = completion['best_step'] if role == 'best' else completion['steps']
    if step != expected_step or not 0 < step <= completion['steps']:
        raise ValueError('Checkpoint embedded step differs from completion metadata')
    if (model.training or any(parameter.requires_grad for parameter in model.parameters())
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError('The loaded model must be frozen/eval with both TF32 switches disabled')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    saved_json_path = provenance_path.parent / 'validation' / ('step%07d.json' % step)
    saved_json = json.loads(saved_json_path.read_text())
    if saved_json['step'] != step or saved_json['split'] != 'world_validation':
        raise ValueError('Saved validation step/split mismatch')
    report.update(checkpoint=str(checkpoint), checkpoint_role=role, step=step, world_dependency=summary,
        training_budget=dict(completed_steps=completion['steps'], configured_steps=completion['configured_steps'],
                             budget_override=completion['budget_override']),
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                     torch_threads=torch.get_num_threads(), amp=False, precision='float32',
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=False),
        datasets={}, comparisons={})
    # All original and executing dependencies are hashed; data arrays remain restricted to validation.
    hashes = {str(path.resolve()): sha(path) for path in artifacts + [saved_json_path, Path(__file__).resolve()]}
    for relative, expected in summary['source_sha256'].items():
        source = (P / relative).resolve(); source.relative_to(P)
        if sha(source) != expected: raise ValueError('Executing source hash changed after strict loading')
        hashes[str(source)] = expected
    for kind, (directory, manifest_path, order) in inspections.items():
        hashes[str(manifest_path)] = sha(manifest_path)
        for item in order: hashes[str((directory / item['file']).resolve())] = item['sha256']
        prediction_path = provenance_path.parent / 'validation' / ('%s_step%07d.npz' % (kind, step))
        hashes[str(prediction_path)] = sha(prediction_path)
        report['datasets'][kind] = dict(directory=str(directory), manifest_sha256=hashes[str(manifest_path)],
            file_order=order, predictions=str(prediction_path), prediction_sha256=hashes[str(prediction_path)])
    report['input_and_source_sha256'] = hashes
    initial_state = model_digest(model)
    levels = model.quantile_levels.cpu().numpy()
    for kind in ('natural', 'paired'):
        prediction_path = Path(report['datasets'][kind]['predictions'])
        report['comparisons'][kind] = verify_dataset(model, prediction_path, inspections[kind], kind == 'paired', batch_size, levels)
    final_state = model_digest(model)
    report['loaded_state_sha256_before'] = initial_state
    report['loaded_state_sha256_after'] = final_state
    if initial_state != final_state or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Frozen model state or gradient contract changed during verification')
    changed = [path for path, expected in hashes.items() if sha(Path(path)) != expected]
    report['inputs_and_sources_unchanged_at_end'] = not changed
    if changed: raise ValueError('Input/source files changed during verification: ' + repr(changed))
    report['passed'] = all(result['passed'] for result in report['comparisons'].values())
    report['status'] = 'passed' if report['passed'] else 'failed_numerical_comparison'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to((R / 'checks').resolve())
    if output.exists(): raise FileExistsError(str(output))
    started = time.time()
    report = dict(schema=1, status='started', passed=False, evidence_kind='full_saved_prediction_recomputation',
        requested_checkpoint=str(args.checkpoint.resolve()), split='world_validation',
        fixed_absolute_tolerances=dict(cgm_mg_dl=CGM_ATOL, bg_class_probability=PROBABILITY_ATOL),
        relative_tolerance=0., all_natural_and_paired_rows_required=True, sampled_comparison=False,
        byte_identity_required=False, checkpoint_selection_performed=False, probability_recalibration=False,
        original_artifacts_modified=False,
        verification_scope='Numerical checkpoint-to-prediction identity only; not independent accuracy, calibration or control validation.')
    try:
        global np, torch
        import numpy as np
        import torch
        run_verification(args.checkpoint, report)
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error))
    report['seconds'] = time.time() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], passed=report['passed'], output=str(output), sha256=sha(output))), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
