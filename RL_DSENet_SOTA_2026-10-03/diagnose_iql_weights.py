"""Read-only full-training-replay weight diagnostics for fixed-final IQL.

No optimizer, sampling, development labels, checkpoint selection or training.
Retain only three FP32 vectors; inference is streamed in ascending replay order.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent
PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inside(path, root):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    return path


def binding(checkpoint, required_sources):
    """Validate metadata and every consumed file before Torch deserialization."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        raise ValueError('Checkpoint must be absolute')
    checkpoint = inside(checkpoint, R/'results')
    if checkpoint.name != 'policy_020000.pt':
        raise ValueError('Only the frozen final 20000-update checkpoint is accepted')
    run = checkpoint.parent
    template = R/'configs/iql_wide.json'
    config = read(run/'config.json'); provenance = read(run/'provenance.json')
    completion = read(run/'completion.json')
    if config != read(template):
        raise ValueError('Saved run configuration differs from its frozen template')
    if (config['name'] != 'IQL_wide' or config['seed'] != 260915 or config['updates'] != 20000
            or config['batch_size'] != 256 or config['state_dim'] != 1613
            or config['advantage_temperature'] != 3 or config['advantage_weight_clip'] != 100):
        raise ValueError('Unexpected fixed IQL contract')
    if (provenance['smoke'] is not False or provenance['training_seed'] != 260915
            or provenance['planned_updates'] != 20000 or provenance['episodes'] != 280
            or provenance['config_sha256'] != sha(template)):
        raise ValueError('Training provenance/configuration mismatch')
    if (completion['status'] != 'fixed_budget_completed' or completion['steps'] != 20000
            or completion['seed'] != 260915 or completion['sample_visits'] != 20000*256
            or completion['development_or_confirmation_used'] is not False
            or completion['final_checkpoint'] != checkpoint.name):
        raise ValueError('Expected completed fixed-budget training without held-out selection')
    records = [item for item in completion['checkpoints'] if item['path'] == checkpoint.name]
    if len(records) != 1 or records[0]['step'] != 20000 or records[0]['sha256'] != sha(checkpoint):
        raise ValueError('Final checkpoint hash/step differs from completion')
    sources = provenance['source_sha256']
    required = set(str(inside(p, P).relative_to(P)) for p in required_sources+[template])
    if not required.issubset(sources):
        raise ValueError('Provenance omits a required IQL/source dependency')
    bound = {}
    for relative, expected in sources.items():
        path = inside(P/relative, P)
        if Path(relative).is_absolute() or sha(path) != expected:
            raise ValueError('Training source hash mismatch: '+relative)
        bound[str(path.relative_to(P))] = expected
    replay = inside(P/provenance['replay_path'], R/'data')
    if replay != inside(R/config['replay'], R/'data'):
        raise ValueError('Checkpoint/replay directory mismatch')
    manifest_path = replay/'manifest.json'; manifest = read(manifest_path)
    if sha(manifest_path) != provenance['replay_manifest_sha256']:
        raise ValueError('Replay manifest hash mismatch')
    if (manifest['status'] != 'complete' or manifest['config'] != config
            or manifest['config_sha256'] != provenance['config_sha256']
            or manifest['no_development_or_confirmation_data'] is not True
            or manifest['identifiers_metadata_only'] is not True
            or manifest['true_BG_reward_only'] is not True):
        raise ValueError('Expected bound training-only replay')
    for left, right in [('source_sha256', 'source_sha256'), ('raw_data_sha256', 'raw_data_sha256'),
                        ('selection', 'data_selection'), ('episodes', 'episodes'), ('transitions', 'transitions')]:
        if manifest[left] != provenance[right]:
            raise ValueError('Replay/training provenance mismatch: '+left)
    if (manifest['natural_episodes'], manifest['ppo_episodes']) != (120, 160):
        raise ValueError('Unexpected training episode mixture')
    if sha(replay/'episodes.json') != manifest['episodes_sha256']:
        raise ValueError('Replay episode metadata hash mismatch')
    required_arrays = {'frames.npy', 'physiology.npy', 'start.npy', 'anchor.npy',
                       'action.npy', 'reward.npy', 'terminal.npy'}
    if not required_arrays.issubset(manifest['array_sha256']):
        raise ValueError('Missing required training replay arrays')
    for filename, expected in manifest['array_sha256'].items():
        if Path(filename).name != filename or not filename.endswith('.npy'):
            raise ValueError('Replay arrays must be local NPY files')
        path = inside(replay/filename, replay)
        if sha(path) != expected:
            raise ValueError('Replay array hash mismatch: '+filename)
        bound[str(path.relative_to(P))] = expected
    for path in [checkpoint, run/'config.json', run/'provenance.json', run/'completion.json',
                 manifest_path, replay/'episodes.json', template]:
        bound[str(path.relative_to(P))] = sha(path)
    return config, provenance, manifest, replay, bound


def validate_checkpoint(state, config, provenance):
    if (state['schema'] != 1 or state['kind'] != 'IQL_wide' or state['step'] != 20000
            or state['smoke'] is not False or state['final'] is not True
            or state['config'] != config or state['provenance'] != provenance):
        raise ValueError('Checkpoint payload differs from its bound final training run')


def statistics(advantage, scaled, weight):
    """Exact retained-vector quantiles; float64 aggregation of FP32 model values."""
    a, z, w = [np.asarray(x, dtype=np.float64) for x in (advantage, scaled, weight)]
    if a.ndim != 1 or a.shape != z.shape or a.shape != w.shape:
        raise ValueError('Expected matching one-dimensional diagnostic vectors')
    if (not np.isfinite(a).all() or not np.isfinite(z).all() or not np.isfinite(w).all()
            or (w < 0).any() or (w > 100.0001).any()):
        raise ValueError('Invalid or nonfinite advantages/weights')
    n = len(a)
    def distribution(x):
        return dict(mean=float(x.mean()) if n else None, std=float(x.std()) if n else None,
                    quantiles={str(p): float(v) for p, v in zip(PERCENTILES,
                               np.quantile(x, np.array(PERCENTILES)/100., method='linear'))} if n else None)
    total = float(w.sum()); squared = float(np.square(w).sum())
    ess = total*total/squared if squared else None
    k = int(math.ceil(.01*n)) if n else 0
    top = float(np.partition(w, n-k)[n-k:].sum()) if k else 0.
    # Match FP32 clamp comparison; exp(float32(log(100))) may round slightly above 100.
    clip_threshold = float(np.float32(math.log(100.)))
    return dict(count=n, advantage=distribution(a), scaled_advantage=distribution(z),
                weight=distribution(w), positive_advantage_fraction=float((a>0).mean()) if n else None,
                weight_clip_count=int((z>=clip_threshold).sum()),
                weight_clip_rate=float((z>=clip_threshold).mean()) if n else None,
                weight_zero_count=int((w==0).sum()), weight_sum=total, effective_sample_size=ess,
                ess_fraction=ess/n if n and ess is not None else None,
                top_one_percent_count=k, top_one_percent_weight_mass=top/total if total else None)


def groups(arrays, episodes, n):
    """Only stored labels/metadata form strata; nothing is inferred from CGM."""
    action = np.asarray(arrays['action'])
    if action.shape != (n,) or not np.isfinite(action).all() or (abs(action)>1).any():
        raise ValueError('Invalid stored normalized action')
    result = {'action': {'[-1,-0.5)': action<-.5, '[-0.5,0)': (action>=-.5)&(action<0),
                         '[0,0.5)': (action>=0)&(action<.5), '[0.5,1]': action>=.5}}
    availability = dict(action='stored action.npy = delivered basal / observed anchor - 1')
    if 'bg_mg_dl' in arrays:
        bg = np.asarray(arrays['bg_mg_dl'])
        if bg.shape != (n,) or not np.isfinite(bg).all() or (bg<=0).any():
            raise ValueError('Invalid stored true BG label')
        result['post_action_bg'] = {'<54': bg<54, '[54,70)': (bg>=54)&(bg<70),
                                   '[70,180]': (bg>=70)&(bg<=180), '>180': bg>180}
        availability['post_action_bg'] = 'stored bg_mg_dl.npy label; never a model input'
    else:
        availability['post_action_bg'] = 'unavailable: no bound bg_mg_dl.npy'
    if 'episode_id' in arrays and all('source' in e for e in episodes):
        ids = np.asarray(arrays['episode_id'])
        if ids.shape != (n,) or ids.dtype.kind not in 'iu':
            raise ValueError('Invalid stored episode IDs')
        mapping = {e['episode_id']: e['source'] for e in episodes}
        if len(mapping) != len(episodes) or not set(np.unique(ids)).issubset(mapping):
            raise ValueError('Episode IDs are missing/duplicated in stored metadata')
        if set(mapping.values()) != {'natural', 'ppo_first8'}:
            raise ValueError('Unexpected training source metadata')
        result['source'] = {name: np.isin(ids, [i for i, source in mapping.items() if source==name])
                            for name in sorted(set(mapping.values()))}
        availability['source'] = 'stored episode_id.npy joined to episodes.json source'
    else:
        availability['source'] = 'unavailable: stored episode_id/source is missing'
    return result, availability


def tensor_digest(agent):
    value = hashlib.sha256()
    for name, tensor in sorted(agent.state_dict().items()):
        value.update(name.encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def run(checkpoint, device, batch_size):
    if not 1 <= batch_size <= 4096:
        raise ValueError('Batch size must be between 1 and 4096')
    from prepare_iql_replay import Replay, load_config, source_paths, history_windows, encode_observations
    config = load_config((R/'configs/iql_wide.json').resolve())
    config, provenance, manifest, replay_path, bound = binding(checkpoint, source_paths(config))
    diagnostic_source = Path(__file__).resolve()
    diagnostic_digest = sha(diagnostic_source)
    bound[str(diagnostic_source.relative_to(P))] = diagnostic_digest
    import torch
    from iql_wide import IQL
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(state, config, provenance)
    replay = Replay(replay_path, config); arrays = replay.arrays; n = replay.size
    if n != manifest['transitions'] or n < 1:
        raise ValueError('Replay transition count mismatch')
    episodes = read(replay_path/'episodes.json')
    if len(episodes) != 280:
        raise ValueError('Expected all 280 stored training episodes')
    starts = np.asarray(arrays['start'])
    if (starts.shape != (n,) or starts.dtype.kind not in 'iu' or starts.min()<71
            or starts.max()+1>=len(arrays['frames'])):
        raise ValueError('Invalid state/next-state frame indices')
    cursor = 0
    for episode in episodes:
        lo, hi = episode['transition_start'], episode['transition_end_exclusive']
        if (lo != cursor or hi<=lo or hi>n or (starts[lo:hi]<episode['frame_start']+71).any()
                or (starts[lo:hi]+1>=episode['frame_end_exclusive']).any()):
            raise ValueError('Replay episode boundaries are invalid')
        if 'episode_id' in arrays and not (arrays['episode_id'][lo:hi]==episode['episode_id']).all():
            raise ValueError('Stored episode IDs disagree with transition boundaries')
        cursor = hi
    if cursor != n:
        raise ValueError('Episode metadata does not cover the full replay')
    strata, availability = groups(arrays, episodes, n)
    python_rng = random.getstate(); numpy_rng = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()] if devices else []
    a = np.empty(n, dtype=np.float32); z = np.empty_like(a); w = np.empty_like(a)
    with torch.random.fork_rng(devices=devices):
        agent = IQL(config).to(device).eval().requires_grad_(False)
        agent.load_state_dict(state['model'], strict=True)
        if not all(torch.isfinite(x).all() for x in agent.state_dict().values()):
            raise ValueError('Checkpoint contains nonfinite model parameters')
        before = tensor_digest(agent)
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                stop = min(start+batch_size, n); ends = starts[start:stop]
                encoded = encode_observations(history_windows(arrays['frames'], ends),
                    arrays['anchor'][start:stop], config, arrays['physiology'][ends])
                x = torch.as_tensor(encoded, device=device)
                action = torch.as_tensor(np.array(arrays['action'][start:stop, None]), device=device)
                pair = torch.cat([x, action], -1)
                advantage = torch.minimum(agent.q1_target(pair), agent.q2_target(pair))-agent.value(x)
                scaled = config['advantage_temperature']*advantage
                weight = scaled.clamp_max(math.log(config['advantage_weight_clip'])).exp()
                a[start:stop] = advantage.flatten().cpu().numpy()
                z[start:stop] = scaled.flatten().cpu().numpy()
                w[start:stop] = weight.flatten().cpu().numpy()
        if tensor_digest(agent) != before or agent.training or any(p.requires_grad or p.grad is not None for p in agent.parameters()):
            raise ValueError('Read-only model-state invariant violated')
    numpy_after = np.random.get_state()
    if (random.getstate()!=python_rng or numpy_after[0]!=numpy_rng[0]
            or not np.array_equal(numpy_after[1], numpy_rng[1]) or numpy_after[2:]!=numpy_rng[2:]
            or not torch.equal(cpu_rng, torch.get_rng_state())
            or any(not torch.equal(x,y) for x,y in zip(cuda_rng, torch.cuda.get_rng_state_all() if devices else []))):
        raise ValueError('Diagnostic changed random-generator state')
    result = dict(schema=1, status='completed_training_only_diagnostic', checkpoint_step=20000, training_seed=260915,
        transitions=n, episodes=len(episodes), batch_size=batch_size,
        inference=dict(device=device, python=sys.version, torch=torch.__version__, numpy=np.__version__,
            dtype='float32', aggregation_dtype='float64', matmul_tf32=False, cudnn_tf32=False, threads=2),
        definition=dict(advantage='min(final target Q1, final target Q2) - final V at stored (s,a)',
            multiplier=3, weight='exp(min(3*A, log(100))) computed in FP32',
            quantile_method='linear, exact retained full vectors; no sampling',
            top_mass_count='ceil(0.01 * stratum count)',
            ess='(sum w)^2 / sum(w^2); concentration diagnostic, not independent-sample count',
            action_strata='stored normalized action, not raw U/h',
            no_last_update_reconstruction=True),
        all_training=statistics(a,z,w),
        strata={kind: {name: statistics(a[mask],z[mask],w[mask]) for name,mask in rows.items()}
                for kind,rows in strata.items()},
        stratum_availability=availability, input_and_source_sha256=bound,
        diagnostic_source_sha256=diagnostic_digest,
        invariants=dict(model_parameters_unchanged=True, model_eval_frozen=True, rng_unchanged=True,
            optimizer_created=False, optimizer_step=False, checkpoint_selection=False,
            data_order='each training transition once in ascending index order',
            development_or_confirmation_read=False),
        limitations=['Final frozen-model diagnostics differ from moving-model training-log window means.',
            'Source/action/BG marginal strata are descriptive, not causal or independent samples.',
            'No gradient comparison, matched BC training, performance claim or automatic retuning.'])
    # Recheck every bound artifact after inference; no input is allowed to drift.
    for relative, expected in bound.items():
        if sha(P/relative) != expected:
            raise ValueError('Bound input/source changed during diagnosis: '+relative)
    result['invariants']['all_bound_artifacts_unchanged'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    output = inside(args.output if args.output.is_absolute() else R/args.output, R/'checks')
    if output.suffix != '.json':
        raise ValueError('Output must be a new checks JSON file')
    # Reserve before work; failure is evidence too, never a silent overwrite/retry.
    with output.open('x') as stream:
        try:
            global np
            import numpy as np
            result = run(args.checkpoint, args.device, args.batch_size)
        except Exception as error:
            result = dict(status='technical_failure', error=repr(error), traceback=traceback.format_exc(),
                          checkpoint=str(args.checkpoint), diagnostic_source_sha256=sha(Path(__file__).resolve()),
                          training_executed=False, development_or_confirmation_read=False)
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
            raise
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], transitions=result['transitions'], output=str(output))))


if __name__ == '__main__':
    main()
