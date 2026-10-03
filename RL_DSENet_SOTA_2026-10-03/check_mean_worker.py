r"""Actual-checkpoint CUDA mechanics for wide/world probability-mean deployment.

Use native Python 3.8 separately for each family; --record is one existing
collect.py world_validation episode JSON with sibling manifest.json/summary.json:
  .venv-native/bin/python <R>/check_mean_worker.py --family wide \
    --config <ABS_RUN>/config.json --checkpoint <ABS_RUN>/policy_iter16.pt \
    --record <ABS_VALIDATION_COLLECTION>/bolus_0.8_p01_s103011.json \
    --output <ABS_R>/checks/mean_wide16_mechanics.json
For world, provide its already completed smoke policy/config explicitly. This
tool checks mechanics only; the separate eval gate authorizes formal 8/16/32/40.
It does not simulate, train, select a checkpoint, or modify original artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

R = Path(__file__).resolve().parent
P = R.parent
TOLERANCES = dict(independent_probability=1e-12, independent_mean_u_h=1e-12,
                  original_argmax_action_u_h=2e-6, original_fp32_probability=2e-7,
                  repeat_logits=1e-5, repeat_probability=1e-6, repeat_action_u_h=1e-6,
                  batch_logits=1e-4, batch_probability=1e-5, batch_action_u_h=2e-5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def model_sha(modules):
    digest = hashlib.sha256()
    for module_name, module in modules.items():
        for name, value in module.state_dict().items():
            raw = value.detach().cpu().contiguous().numpy()
            digest.update((module_name + '.' + name + str(raw.shape) + str(raw.dtype)).encode('utf-8'))
            digest.update(raw.tobytes())
    return digest.hexdigest()


def independent_binding(config, checkpoint):
    config = config.resolve(); checkpoint = checkpoint.resolve()
    run = config.parent; run.relative_to((R / 'results').resolve())
    if config.name != 'config.json' or checkpoint.parent != run:
        raise ValueError('Expected same-run config.json and policy checkpoint')
    name = checkpoint.stem
    if not name.startswith('policy_iter') or not name[len('policy_iter'):].isdigit():
        raise ValueError('Expected explicit policy_iter checkpoint')
    iteration = int(name[len('policy_iter'):])
    c = json.loads(config.read_text()); provenance = json.loads((run / 'provenance.json').read_text())
    if (checkpoint.name != 'policy_iter%02d.pt' % iteration or not 1 <= iteration <= c['iterations']
            or c['seed'] != 260915 or c['name'] != run.name
            or provenance['config'] != c or provenance['training_seed'] != 260915):
        raise ValueError('Policy config/provenance/iteration mismatch')
    lines = []
    with (run / 'history.jsonl').open('rb') as stream:
        for i in range(1, iteration + 1):
            line = stream.readline()
            if not line.endswith(b'\n'): raise ValueError('Incomplete training-history prefix')
            item = json.loads(line)
            if type(item['iteration']) is not int or item['iteration'] != i:
                raise ValueError('Noncontiguous completed training-history prefix')
            lines.append(line)
    if (not Path(item['checkpoint']).is_absolute() or Path(item['checkpoint']).resolve() != checkpoint
            or item['checkpoint_sha256'] != sha(checkpoint)):
        raise ValueError('Checkpoint SHA does not match its completed training record')
    prefix = b''.join(lines)
    return dict(iteration=iteration, checkpoint_sha256=sha(checkpoint), config_sha256=sha(config),
                training_provenance_sha256=sha(run / 'provenance.json'),
                history_prefix_sha256=hashlib.sha256(prefix).hexdigest(), history_prefix_bytes=len(prefix))


def observed_histories(record):
    """Build real 360/420-minute histories from past observed fields only."""
    record = record.resolve(); record.relative_to((R / 'results').resolve())
    manifest_path = record.parent / 'manifest.json'; summary_path = record.parent / 'summary.json'
    manifest = json.loads(manifest_path.read_text()); summary = json.loads(summary_path.read_text())
    protocol = json.loads((R / 'protocol.json').read_text())
    if manifest['config']['split'] != 'world_validation' or summary['status'] != 'completed':
        raise ValueError('A completed natural world_validation collector record is required')
    if not any(item['key'] == record.stem for item in summary['episodes']):
        raise ValueError('Episode does not belong to the completed collection summary')
    raw = json.loads(record.read_text()); job = raw['job']
    if (job not in manifest['jobs'] or job['seed'] not in protocol['world_validation_scenario_seeds']
            or job['patient'] not in protocol['patients'] or job['bolus_factor'] not in protocol['bolus_factors']
            or raw['failure_reason'] not in (None, 'native_environment_done')):
        raise ValueError('Record job/split/technical-failure gate failed')
    rows = raw['records']
    if len(rows) < 84: raise ValueError('Need real history through minute 420')
    sys.path.insert(0, str(P / 'RL进阶对比_2026-09-15'))
    from observable_history import History
    history = History(); states = []
    for index, row in enumerate(rows[:84]):
        if row['minute'] != (index + 1) * 5: raise ValueError('Expected original five-minute record sequence')
        history.append(row['minute'], row['cgm_mg_dl'], row['delivered_basal_u_h'] / 12,
                       row['bolus_u'] if row['bolus_u'] > 0 else None,
                       row['meal_g'] if row['meal_g'] > 0 else None)
        if index in (71, 83): states.append(history.state())
    histories = np.stack(states).astype(np.float32)
    if histories[0, -1, 6] <= .5: raise ValueError('Warmup anchor must be actually observed')
    anchor = float((histories[0, -1, 1] * .14462788945609448 + .09945811581924525) * 12)
    anchors = np.array([anchor, anchor], dtype=np.float32)
    return histories, anchors, dict(record=str(record), record_sha256=sha(record),
        manifest_sha256=sha(manifest_path), summary_sha256=sha(summary_path),
        origin_minutes=[360, 420], split='world_validation', history_shape=list(histories.shape),
        history_bytes_sha256=hashlib.sha256(histories.tobytes()).hexdigest(), anchors_u_h=anchors.tolist(),
        observed_fields=['cgm_mg_dl', 'delivered_basal_u_h', 'bolus_u', 'meal_g', 'minute'],
        BG_identity_future_meals_sent_to_policy=False)


def maximum_error(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Compared arrays must have matching shape and finite elements')
    return float(np.max(np.abs(first - second)))


def run(args, report):
    from ppo_mean_worker import MeanController, distribution_summary, RATIOS
    config = args.config.resolve(); checkpoint = args.checkpoint.resolve(); record = args.record.resolve()
    binding = independent_binding(config, checkpoint)
    histories, anchors, data_provenance = observed_histories(record)
    paths = [config, checkpoint, config.parent / 'provenance.json', record,
             record.parent / 'manifest.json', record.parent / 'summary.json', R / 'protocol.json',
             R / 'ppo_mean_worker.py', R / 'ppo_wide_worker.py', R / 'ppo_world_worker.py',
             R / 'physiologic_features.py', Path(__file__).resolve(),
             P / 'RL进阶对比_2026-09-15/observable_history.py',
             P / 'Loop数据集/训练管线_v2/prepared/normalization.json']
    hashes = {str(path): sha(path) for path in paths}
    report.update(binding=binding, record=data_provenance, source_and_input_sha256=hashes,
                  errors={}, checks=[], family=args.family)
    def check(name, condition):
        report['checks'].append(dict(name=name, passed=bool(condition)))
        if not condition: raise AssertionError(name)
    def compare(name, first, second, tolerance):
        error = maximum_error(first, second); report['errors'][name] = dict(maximum=error, tolerance=tolerance)
        check(name, error <= tolerance)
    controller = MeanController(args.family, config, checkpoint)
    ready = controller.ready; trainer = controller.trainer
    check('ready_family_interface', ready['ready'] and ready['kind'] == 'ppo_mean' and ready['family'] == args.family
          and ready['deployment_rule'] == 'probability_mean_of_capped_executable_grid')
    check('ready_independent_history_and_sha_binding', all(ready[key] == value for key, value in binding.items()))
    check('ready_no_training_or_sampling', not ready['training_enabled'] and not ready['random_action_sampling']
          and not ready['rollout_buffer_retained'])
    check('ready_source_hashes', all(sha(P / relative) == expected for relative, expected in ready['source_sha256'].items()))
    check('ready_provenance_and_feature_dim', ready['provenance'] == trainer.provenance
          and ready['feature_dim'] == (334 if args.family == 'wide' else 195))
    report.update(ready_sha256=json_sha(ready), runtime=ready['runtime'])
    modules = dict(policy=trainer.policy, value=trainer.value, world=trainer.world)
    check('no_optimizer_or_rollout_buffer', all(not hasattr(trainer, name) for name in ('opt', 'vo', 'buffer')))
    check('all_models_eval_and_frozen', all(not module.training and all(not p.requires_grad and p.grad is None
          for p in module.parameters()) for module in modules.values()))
    state = torch.load(checkpoint, map_location='cpu')
    check('actor_tensors_equal_checkpoint', all(torch.equal(value.cpu(), state['model'][name])
          for name, value in trainer.policy.state_dict().items()))
    check('critic_tensors_equal_checkpoint', all(torch.equal(value.cpu(), state['value'][name])
          for name, value in trainer.value.state_dict().items()))
    del state
    before = model_sha(modules); count_before = trainer.count
    rng_cpu = torch.get_rng_state().clone(); rng_cuda = [value.clone() for value in torch.cuda.get_rng_state_all()]
    rng_python = random.getstate(); rng_numpy = np.random.get_state()
    request = dict(op='evaluate', history=histories.tolist(), anchors=anchors.tolist())
    def observed_evaluate(payload):
        captured = []
        hook = trainer.policy.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().cpu().clone()))
        try: response = controller.evaluate(payload)
        finally: hook.remove()
        check('one_original_actor_forward', len(captured) == 1)
        return response, captured[0].numpy()
    one, actual_logits = observed_evaluate(request)
    two, repeat_logits = observed_evaluate(request)
    with torch.no_grad():
        features, tensor_anchor = trainer.features(histories, anchors)
        logits = trainer.policy(features)
        # Separate aggregation agreement from allowed repeated-forward roundoff.
        captured_tensor = torch.as_tensor(actual_logits, device='cuda')
        probabilities = torch.softmax(captured_tensor.double(), -1).cpu().numpy()
        original_fp32_probabilities = torch.softmax(captured_tensor, -1).cpu().numpy()
        grid = (tensor_anchor.double()[:, None] * trainer.multipliers.double()[None]).clamp(0, 20).cpu().numpy()
        expected_mean = np.sum(probabilities * grid, axis=1)
    compare('wrapper_original_actor_logits', actual_logits, logits.cpu().numpy(), TOLERANCES['repeat_logits'])
    compare('independent_softmax_probability', one['probabilities'], probabilities, TOLERANCES['independent_probability'])
    compare('original_FP32_softmax_difference', one['probabilities'], original_fp32_probabilities, TOLERANCES['original_fp32_probability'])
    compare('independent_executable_grid', one['actual_action_grid_u_h'], grid, 0.)
    compare('independent_grid_probability_mean', one['actions'], expected_mean, TOLERANCES['independent_mean_u_h'])
    compare('action_mean_alias', one['action_mean'], one['actions'], 0.)
    # Invoke only the original evaluation path on the restored trainer; it does not require a buffer.
    original = trainer.act(request, evaluate=True)
    check('original_argmax_indices', original['action_indices'] == one['argmax_indices'])
    compare('original_argmax_rates', original['actions'], one['argmax_actions_u_h'], TOLERANCES['original_argmax_action_u_h'])
    compare('repeat_logits', actual_logits, repeat_logits, TOLERANCES['repeat_logits'])
    compare('repeat_probabilities', one['probabilities'], two['probabilities'], TOLERANCES['repeat_probability'])
    compare('repeat_actions', one['actions'], two['actions'], TOLERANCES['repeat_action_u_h'])
    single_logits = []; single_probabilities = []; single_actions = []; single_argmax = []
    for i in range(2):
        response, captured = observed_evaluate(dict(op='evaluate', history=histories[i:i + 1].tolist(), anchors=anchors[i:i + 1].tolist()))
        single_logits.extend(captured.tolist()); single_probabilities.extend(response['probabilities'])
        single_actions.extend(response['actions']); single_argmax.extend(response['argmax_indices'])
    compare('batch_single_logits', actual_logits, single_logits, TOLERANCES['batch_logits'])
    compare('batch_single_probabilities', one['probabilities'], single_probabilities, TOLERANCES['batch_probability'])
    compare('batch_single_actions', one['actions'], single_actions, TOLERANCES['batch_action_u_h'])
    check('batch_single_argmax_indices', one['argmax_indices'] == single_argmax)
    synthetic_logits = [-1000.] * 9; synthetic_logits[0] = synthetic_logits[-1] = 0.
    boundary = distribution_summary([synthetic_logits], [20.], RATIOS)
    check('synthetic_large_anchor_cap_before_mean', boundary['actions'] == [10.] and boundary['raw_ratio_mean_rate'] == [20.])
    rejected = 0
    for invalid in (dict(request, op='act'), dict(request, op='update'), dict(request, bg=[100., 100.]), dict(request, indices=[0, 1])):
        try: controller.evaluate(invalid)
        except ValueError: rejected += 1
    check('training_and_hidden_fields_rejected', rejected == 4)
    after = model_sha(modules)
    check('all_model_parameters_and_buffers_unchanged', before == after)
    check('all_models_still_frozen_and_no_gradients', all(not module.training and all(not p.requires_grad and p.grad is None
          for p in module.parameters()) for module in modules.values()))
    check('no_training_state_created', trainer.count == count_before and all(not hasattr(trainer, name) for name in ('opt', 'vo', 'buffer')))
    check('CPU_CUDA_RNG_unchanged', torch.equal(rng_cpu, torch.get_rng_state())
          and all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())))
    np_after = np.random.get_state()
    check('Python_NumPy_RNG_unchanged', rng_python == random.getstate() and rng_numpy[0] == np_after[0]
          and np.array_equal(rng_numpy[1], np_after[1]) and rng_numpy[2:] == np_after[2:])
    check('checkpoint_history_prefix_binding_unchanged', independent_binding(config, checkpoint) == binding)
    check('source_and_input_files_unchanged', all(sha(Path(path)) == digest for path, digest in hashes.items()))
    report.update(status='passed', passed=True, model_state_sha256_before=before, model_state_sha256_after=after,
        actual_logits=actual_logits.tolist(), actions=one['actions'], probabilities=one['probabilities'],
        argmax_indices=one['argmax_indices'], argmax_actions_u_h=one['argmax_actions_u_h'],
        actual_action_grid_u_h=one['actual_action_grid_u_h'], count_checks=len(report['checks']),
        real_checkpoint_loaded=True, real_record_history_used=True, synthetic_boundary_case_only=boundary['actions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', choices=('wide', 'world'), required=True)
    for key in ('config', 'checkpoint', 'record', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.relative_to((R / 'checks').resolve())
    if output.exists(): raise FileExistsError(str(output))
    started = time.time()
    report = dict(schema=1, status='started', passed=False, mechanics_only=True, formal_evaluation_authorized=False,
                  model_or_clinical_performance_claim=False, simulator_executed=False, tolerances=TOLERANCES,
                  source_sha256=sha(Path(__file__).resolve()))
    try:
        global np, torch
        import numpy as np
        import torch
        run(args, report)
    except Exception as error:
        report.update(status='failed', passed=False, error=repr(error))
    report['seconds'] = time.time() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], passed=report['passed'], output=str(output), sha256=sha(output))), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
