"""Bindings and pure mechanics for conditional planner BC; no simulator calls."""
import hashlib
import json
import math
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
P = R.parent
SEED = 260915
FIT_SEEDS = (103001, 103002, 103003)
CHECK_SEED = 103004
PLAN_TO_ACTION = (4, 0, 2, 6, 8, 0, 2, 6, 8)
MULTIPLIERS = (0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.)
BC_CONFIG = dict(seed=SEED, epochs=3, batch_size=512, learning_rate=.0003,
                 gradient_clip=.5, label_smoothing=.05, optimizer='Adam',
                 teacher_mode='risk', fit_seeds=list(FIT_SEEDS), check_seed=CHECK_SEED,
                 input_dim=195, action_count=9, overall_agreement_min=.9,
                 nonhold_agreement_min=.8, nonhold_samples_min=100,
                 full_nonhold_episodes_min=3, action_tolerance_u_h=1e-6,
                 precision='float32_no_amp_no_tf32', torch_threads=2)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def simple_name(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('A new simple directory name is required')


def verify_files(bindings):
    for path, digest in bindings.items():
        if sha(path) != digest: raise ValueError('Bound file changed: ' + path)


def snapshot(paths):
    return {str(p.resolve()): sha(p) for p in sorted(set(Path(p) for p in paths))}


def state_sha(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        x = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(x.shape).encode())
        result.update(str(x.dtype).encode()); result.update(x.tobytes())
    return result.hexdigest()


def runtime_seed():
    import random
    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False


def validate_natural_manifest(manifest, protocol, expected_order):
    """Run before NPZ reads; PPO train scenes and all validation scenes are rejected."""
    if (manifest.get('status') != 'completed' or manifest.get('split') != 'train'
            or manifest.get('kind') == 'paired_interventions' or manifest.get('horizon') != 72
            or type(manifest.get('stride')) is not int or manifest['stride'] < 1
            or protocol['training_seed'] != SEED
            or protocol['train_scenario_seeds'] != list(FIT_SEEDS) + [CHECK_SEED]):
        raise ValueError('Only the existing four natural train scenarios are accepted')
    rows = sorted(manifest['episodes'], key=lambda item: item['file'])
    if not rows or len({r['file'] for r in rows}) != len(rows): raise ValueError('Empty/repeated pack file')
    actual = []; jobs = set()
    for row in rows:
        name = Path(row['file']); job = row['job']
        if name.name != str(name) or name.suffix != '.npz': raise ValueError('Expected local NPZ name')
        key = (job['patient'], job['seed'], job['bolus_factor'])
        if (job['seed'] not in FIT_SEEDS + (CHECK_SEED,) or job['patient'] not in protocol['patients']
                or job['bolus_factor'] not in protocol['bolus_factors'] or key in jobs):
            raise ValueError('Train scene/patient/factor mismatch or duplicate episode')
        jobs.add(key)
        actual.append(dict(file=row['file'], count=row['origins'], sha256=row['sha256'], job=job))
    expected_jobs = {(p, s, f) for p in protocol['patients'] for s in FIT_SEEDS + (CHECK_SEED,)
                     for f in protocol['bolus_factors']}
    if jobs != expected_jobs or actual != expected_order:
        raise ValueError('Natural train must match the complete world-bound file order')
    if sum(row['origins'] for row in rows) != manifest['origins']: raise ValueError('Origin count mismatch')
    return rows


def inspect_inputs(natural, world_checkpoint, name):
    """Inspect immutable train metadata and strict completed-world bindings only."""
    simple_name(name)
    if not name.startswith('PPO_world_bc'): raise ValueError('Use an explicit PPO_world_bc... new name')
    natural = Path(natural).resolve(); natural.relative_to((R / 'cache').resolve())
    checkpoint = Path(world_checkpoint)
    if not checkpoint.is_absolute(): raise ValueError('World checkpoint must be absolute')
    checkpoint = checkpoint.resolve()
    from world_control_worker import inspect_provenance, FROZEN_CONFIG
    from train_world_policy import validate_config
    provenance_path, world, sources, artifacts = inspect_provenance(checkpoint)
    completed = read(provenance_path.parent / 'completion.json')
    if (world['config']['variant'] != 'quantile' or completed['steps'] != 4000
            or completed['configured_steps'] != 4000 or completed['budget_override'] is not False):
        raise ValueError('The fixed teacher requires the completed formal quantile world')
    protocol = read(R / 'protocol.json'); pack = read(natural / 'manifest.json')
    bound = world['data']['natural_train']
    if (Path(bound['path']).resolve() != natural or bound['split'] != 'train' or bound['paired'] is not False
            or bound['manifest_sha256'] != sha(natural / 'manifest.json')):
        raise ValueError('Natural pack is not the training data bound to this world')
    rows = validate_natural_manifest(pack, protocol, bound['file_order'])
    simple_name(pack['source']); raw_dir = (R / 'results' / pack['source']).resolve()
    raw_dir.relative_to((R / 'results').resolve())
    raw_manifest = read(raw_dir / 'manifest.json'); raw_summary = read(raw_dir / 'summary.json')
    if (sha(raw_dir / 'manifest.json') != pack['source_manifest_sha256']
            or raw_manifest['config']['split'] != 'train' or raw_manifest['config']['smoke'] is not False
            or raw_manifest['protocol'] != protocol or raw_summary['status'] != 'completed'
            or {row['file'][:-4] for row in rows} != {item['key'] for item in raw_summary['episodes']}
            or len(raw_manifest['jobs']) != len(rows) or len(raw_summary['episodes']) != len(rows)
            or sorted(raw_manifest['jobs'], key=lambda j: (j['patient'], j['seed'], j['bolus_factor']))
            != sorted([r['job'] for r in rows], key=lambda j: (j['patient'], j['seed'], j['bolus_factor']))):
        raise ValueError('Raw collection/pack split, jobs or completion mismatch')
    if pack['source_sha256'] != sha(R / 'prepare_windows.py'): raise ValueError('Packing source changed')
    collector_sources = [R / 'collect.py', R / 'controller_baselines.py', R / 'physiologic_features.py',
                         P / 'RL_DSENet_公平低糖_2026-09-21/ppo_env.py', P / protocol['scorer']]
    if raw_manifest['source_sha256'] != {p.name: sha(p) for p in collector_sources}:
        raise ValueError('Raw collector source binding mismatch')
    c = read(R / 'configs/ppo_world_risk.json')
    c.update(name=name, output_dir=str((R / 'results' / name).resolve()), world_checkpoint=str(checkpoint),
             world_checkpoint_sha256=sha(checkpoint), world_training_provenance_sha256=sha(provenance_path),
             world_completion_sha256=sha(provenance_path.parent / 'completion.json'))
    validate_config(c, protocol, bound=True)
    extra_sources = [R / x for x in ('planner_bc_common.py', 'prepare_planner_bc.py', 'train_planner_bc.py',
        'ppo_world_worker.py', 'ppo_wide_worker.py', 'train_world_policy.py', 'world_control_worker.py',
        'physiologic_features.py', 'protocol.json')]
    extra_sources += [P / 'RL进阶对比_2026-09-15/observable_history.py',
        P / 'RL_DSENet_2026-09-17/world_model.py', P / 'RL_DITR创新_2026-09-16/ditr_model.py',
        P / 'RL_DITR创新_2026-09-16/response_operator.py']
    paths = sources + artifacts + collector_sources + extra_sources + [natural / 'manifest.json',
        raw_dir / 'manifest.json', raw_dir / 'summary.json', R / 'configs/ppo_world_risk.json']
    for row in rows:
        pack_path = natural / row['file']; raw_path = raw_dir / (Path(row['file']).stem + '.json')
        if sha(pack_path) != row['sha256'] or sha(raw_path) != row['source_sha256']:
            raise ValueError('Train pack/raw file SHA mismatch')
        paths += [pack_path, raw_path]
    return dict(policy_config=c, bc_config=BC_CONFIG, teacher_config=dict(FROZEN_CONFIG, mode='risk'),
                natural_path=str(natural), raw_path=str(raw_dir), stride=pack['stride'], records=rows,
                world_provenance_sha256=sha(provenance_path), bound_file_sha256=snapshot(paths),
                source_sha256=snapshot(sources + collector_sources + extra_sources),
                design='conditional_post_development_BC_then_true_PPO_not_activated_by_preparation')


def observed_pack(binding, record):
    """Reconstruct every historical input; never index stored future labels/masks."""
    import numpy as np
    sys.path.insert(0, str(P / 'RL进阶对比_2026-09-15'))
    from observable_history import History
    path = Path(binding['natural_path']) / record['file']
    raw_path = Path(binding['raw_path']) / (Path(record['file']).stem + '.json')
    if sha(path) != record['sha256'] or sha(raw_path) != record['source_sha256']:
        raise ValueError('Pack/raw changed after inspection')
    with np.load(path, allow_pickle=False) as pack:
        h, a, minutes = (pack[k] for k in ('history', 'anchor_u_h', 'origin_minute'))
    n = record['origins']
    if (h.shape != (n, 72, 22) or a.shape != (n,) or minutes.shape != (n,) or n < 1
            or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (h, a, minutes))
            or np.any((a <= 0) | (a > 20))): raise ValueError('Observed input shape/dtype/value mismatch')
    raw = read(raw_path)
    if raw['job'] != record['job'] or raw['failure_reason'] not in (None, 'native_environment_done'):
        raise ValueError('Raw job or technical failure mismatch')
    rows = raw['records']; origins = list(range(71, len(rows) - 1, binding['stride']))
    if len(origins) != n: raise ValueError('Pack omitted/added origins')
    history = History(); wanted = set(origins); index = 0
    anchor = np.float32(rows[71]['delivered_basal_u_h'])
    if not np.array_equal(a, np.full(n, anchor, dtype=np.float32)): raise ValueError('Anchor mismatch')
    for i, row in enumerate(rows):
        if i > origins[-1]: break
        history.append(row['minute'], row['cgm_mg_dl'], row['delivered_basal_u_h'] / 12,
                       row['bolus_u'] if row['bolus_u'] > 0 else None,
                       row['meal_g'] if row['meal_g'] > 0 else None)
        if i in wanted:
            if not np.array_equal(history.state(), h[index]) or np.float32(row['minute']) != minutes[index]:
                raise ValueError('Observed history/origin differs from raw prefix')
            index += 1
    full = raw['failure_reason'] is None and rows[-1]['minute'] == raw['job']['total_minutes']
    return h, a, minutes, bool(full)


def map_teacher(chosen, plans, anchors):
    import numpy as np
    chosen = np.asarray(chosen); anchors = np.asarray(anchors, dtype=np.float32)
    if chosen.shape != anchors.shape or not np.issubdtype(chosen.dtype, np.integer) or np.any((chosen < 0) | (chosen > 8)):
        raise ValueError('Invalid teacher plan index')
    labels = np.asarray(PLAN_TO_ACTION, dtype=np.int64)[chosen]
    grid = np.clip(anchors[:, None] * np.asarray(MULTIPLIERS, dtype=np.float32), 0, 20)
    actions = np.asarray(plans)[np.arange(len(chosen)), chosen, 0]
    if not np.array_equal(actions, grid[np.arange(len(chosen)), labels]):
        raise ValueError('Teacher plan does not map exactly to the original executable actor grid')
    return labels, actions.astype(np.float32)


def agreement_gate(overall, nonhold, nonhold_count, full_nonhold_episodes):
    if (not math.isfinite(overall) or not 0 <= overall <= 1
            or (nonhold is not None and (not math.isfinite(nonhold) or not 0 <= nonhold <= 1))
            or type(nonhold_count) is not int or nonhold_count < 0
            or type(full_nonhold_episodes) is not int or full_nonhold_episodes < 0):
        raise ValueError('Invalid agreement/count diagnostic')
    reasons = []
    if overall < BC_CONFIG['overall_agreement_min']: reasons.append('overall_agreement_below_0.90')
    if nonhold is None or nonhold < BC_CONFIG['nonhold_agreement_min']: reasons.append('nonhold_agreement_below_0.80')
    if nonhold_count < BC_CONFIG['nonhold_samples_min']: reasons.append('fewer_than_100_nonhold_states')
    if full_nonhold_episodes < BC_CONFIG['full_nonhold_episodes_min']: reasons.append('fewer_than_3_full_nonhold_source_episodes')
    return dict(passed=not reasons, reasons=reasons, clinical_validation=False)


def inspect_cache(directory):
    directory = Path(directory).resolve(); directory.relative_to((R / 'cache').resolve())
    manifest = read(directory / 'manifest.json')
    if (manifest['status'] != 'prepared_no_training' or manifest['bc_config'] != BC_CONFIG
            or (directory / 'failure.json').exists()
            or manifest['world_state_before'] != manifest['world_state_after']
            or manifest['actor_state_before'] != manifest['actor_state_after']
            or manifest['value_state_before'] != manifest['value_state_after']):
        raise ValueError('Invalid/incomplete frozen feature cache')
    verify_files(manifest['binding']['bound_file_sha256'])
    if (read(directory / 'policy_config.json') != manifest['binding']['policy_config']
            or sha(directory / 'policy_config.json') != manifest['policy_config_sha256']
            or sha(directory / 'input_binding.json') != manifest['input_binding_sha256']
            or read(directory / 'input_binding.json') != manifest['binding']):
        raise ValueError('Cache policy config mismatch')
    expected = manifest['binding']['records']; rows = manifest['files']
    if len(rows) != len(expected): raise ValueError('Cache episode coverage mismatch')
    for item, source in zip(rows, expected):
        if (item['file'] != source['file'] or item['job'] != source['job'] or item['count'] != source['origins']
                or item['part'] != ('fit' if source['job']['seed'] in FIT_SEEDS else 'internal_check')
                or sha(directory / item['file']) != item['sha256']):
            raise ValueError('Feature cache coverage/order/SHA mismatch')
    return directory, manifest
