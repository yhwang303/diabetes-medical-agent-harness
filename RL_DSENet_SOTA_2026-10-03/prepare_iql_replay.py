"""Freeze 120 natural plus the first 160 PPO training episodes for IQL.

No simulator/model is imported or run. Histories contain observed records only;
BG is a reward label. Compact frame indices preserve every episode boundary.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
HISTORY_SOURCE = P/'RL进阶对比_2026-09-15/observable_history.py'
SCORER = P/'RL_DITR创新_2026-09-16/control_metrics.py'
UPSTREAM = R/'vendor/iql'
sys.path.insert(0, str(HISTORY_SOURCE.parent))
sys.path.insert(0, str(SCORER.parent))
from observable_history import History
from control_metrics import summarize
from physiologic_features import FEATURE_DIM, FEATURE_NAMES, NORMALIZER_PATH, features, normalization

CLOCK = np.stack([np.ones(72)/12, np.arange(-71, 1)/12], -1).astype(np.float32)
OFFSETS = np.arange(-71, 1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('--config requires an absolute path')
    c = read_json(path)
    if (c['seed'], c['updates'], c['batch_size'], c['state_dim']) != (260915, 20000, 256, 1613):
        raise ValueError('Frozen IQL seed/budget/input contract changed')
    if FEATURE_DIM != 28 or c['actor_lr_schedule'] != 'cosine' or c['reward_profile'] != 'risk':
        raise ValueError('Feature or IQL schedule contract changed')
    if (c['natural_name'], c['natural_episodes'], c['ppo_name'], c['ppo_iterations'],
        c['ppo_cases_per_iteration'], c['ppo_episodes']) != ('natural_train_r1', 120, 'PPO_wide_risk', 8, 20, 160):
        raise ValueError('Data selection must remain the preregistered 120+160 episodes')
    if (c['gamma'], c['low_risk_weight'], c['reward_scale'], c['terminal_penalty']) != (.997, 2., 10., 100.):
        raise ValueError('Frozen PPO-matched reward/discount changed')
    if c['pump_rounding_tolerance_u_h'] > .0002501 or c['request_roundoff_tolerance_u_h'] > 1e-6:
        raise ValueError('Pump tolerance cannot hide a policy projection')
    return c


def history_windows(frames, ends):
    observed = np.asarray(frames[ends[:, None]+OFFSETS], dtype=np.float32)
    return np.concatenate([observed, np.broadcast_to(CLOCK, (len(ends), 72, 2))], axis=-1)


def encode_observations(history, anchors, config, physiological=None):
    h = np.asarray(history, dtype=np.float32)
    a = np.asarray(anchors, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (72, 22) or a.shape != (len(h),):
        raise ValueError('Expected history[N,72,22] and anchors[N]')
    if not np.isfinite(h).all() or not np.isfinite(a).all() or (a <= 0).any() or (a > 20).any():
        raise ValueError('Nonfinite observed history or invalid anchor')
    physical = features(h, a) if physiological is None else np.asarray(physiological, dtype=np.float32)
    if physical.shape != (len(h), FEATURE_DIM) or not np.isfinite(physical).all():
        raise ValueError('Invalid physiology feature shape/value')
    return np.concatenate([h.reshape(len(h), -1), physical,
                           (a/config['anchor_scale'])[:, None]], axis=-1).astype(np.float32)


def risk_reward(bg, terminal, config):
    """Same un-clipped Kovatchev risk and terminal penalty as wide PPO."""
    bg = np.asarray(bg, dtype=np.float64)
    if bg.ndim != 1 or not len(bg) or not np.isfinite(bg).all() or (bg <= 0).any():
        raise ValueError('A finite positive true-BG label is required')
    f = 1.509*(np.log(np.maximum(bg, 1.)).clip(min=0)**1.084-5.381)
    risk = 10.*f**2
    high, low = np.where(f >= 0, risk, 0.), np.where(f < 0, risk, 0.)
    reward = -(high+config['low_risk_weight']*low)/(12.*config['reward_scale'])
    if terminal:
        reward[-1] -= config['terminal_penalty']
    return reward


def episode_arrays(raw, config, require_complete=False):
    """t360 state -> delivered action -> t365 BG/reconstructed next state."""
    failure = raw['failure_reason']
    if failure not in (None, 'native_environment_done'):
        raise ValueError('Technical/nonfinite failure is not a physiological terminal: '+str(failure))
    terminal = failure == 'native_environment_done'
    records = raw['records']; total = raw['job']['total_minutes']
    if total != 4320 or len(records) <= 72 or len(records) > total//5:
        raise ValueError('Wrong duration or no post-warmup transitions')
    if (not terminal or require_complete) and (terminal or len(records) != total//5):
        raise ValueError('Required full episode is incomplete')
    history = History()
    for i, row in enumerate(records):
        if row['minute'] != (i+1)*5 or row['warmup'] != (i < 72):
            raise ValueError('Shifted/missing row or warmup label')
        values = [row[k] for k in ('cgm_mg_dl', 'bg_mg_dl', 'delivered_basal_u_h',
                                   'requested_basal_u_h', 'bolus_u', 'meal_g')]
        if any(v is None for v in values) or not np.isfinite(values).all() or min(values) < 0 or min(values[:2]) <= 0:
            raise ValueError('Incomplete or invalid observed transition')
        history.append(row['minute'], row['cgm_mg_dl'], row['delivered_basal_u_h']/12,
                       row['bolus_u'] if row['bolus_u'] > 0 else None,
                       row['meal_g'] if row['meal_g'] > 0 else None)
    frames = np.asarray(history.rows, dtype=np.float32)
    first = history_windows(frames, np.array([71]))
    mean, scale, _ = normalization()
    anchor = float((float(first[0, -1, 1])*scale[1]+mean[1])*12)
    if not 0 < anchor <= 20 or abs(anchor-records[71]['delivered_basal_u_h']) > 1e-6:
        raise ValueError('Warmup anchor not recovered from actual observed basal')
    rows = records[72:]
    requested = np.array([x['requested_basal_u_h'] for x in rows])
    delivered = np.array([x['delivered_basal_u_h'] for x in rows])
    upper = min(20., 2*anchor)
    request_tolerance = config['request_roundoff_tolerance_u_h']
    rounding_tolerance = config['pump_rounding_tolerance_u_h']
    if (requested < 0).any() or (requested > upper+request_tolerance).any():
        raise ValueError('Raw policy request exceeds the common action range')
    if np.max(np.abs(delivered-requested)) > rounding_tolerance:
        raise ValueError('Delivered/requested difference exceeds verified pump rounding')
    # Check the physical pump rule, independently of normalized action clipping.
    expected_pump = np.round(requested/60*6000/.05)*.05/6000*60
    if not np.allclose(delivered, expected_pump, atol=1e-8, rtol=0):
        raise ValueError('Delivered actions differ from frozen Insulet quantization')
    raw_action = delivered/anchor-1
    excess = np.maximum(delivered-upper, 0)
    if (delivered < 0).any() or excess.max() > rounding_tolerance+request_tolerance:
        raise ValueError('Delivered action exceeds explainable boundary roundoff')
    action = np.clip(raw_action, -1, 1)
    if anchor > 10 and (delivered > 20).any():
        raise ValueError('Global physical cap exceeded')
    # Positive anchors in this frozen adult task are <10. Enforce the upper cap
    # separately at inference; normalized [-1,1] remains the learned coordinate.
    ends = np.arange(71, len(records))
    windows = history_windows(frames, ends)
    physiological = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    physiological[ends] = features(windows, np.full(len(ends), anchor))
    n = len(rows); terminal_flags = np.zeros(n, dtype=bool); terminal_flags[-1] = terminal
    time_limit = np.zeros(n, dtype=bool); time_limit[-1] = not terminal
    bg = np.array([x['bg_mg_dl'] for x in rows])
    metrics = summarize(records, (total-360)//5, terminal)
    if metrics != raw['metrics']:
        raise ValueError('Raw scores differ from the unchanged scorer; use original numeric runtime')
    arrays = dict(frames=frames, physiology=physiological, start=np.arange(71, len(records)-1, dtype=np.int64),
                  anchor=np.full(n, anchor, dtype=np.float64), action=action.astype(np.float32),
                  reward=risk_reward(bg, terminal, config).astype(np.float32), terminal=terminal_flags,
                  time_limit=time_limit, bg_mg_dl=bg.astype(np.float32),
                  delivered_u_h=delivered.astype(np.float64), requested_u_h=requested.astype(np.float64))
    changed = np.flatnonzero(action != raw_action)
    audit = dict(anchor_u_h=anchor, warmup_delivered_u_h=records[71]['delivered_basal_u_h'],
                 terminal=terminal, time_limit=not terminal, records=len(records), transitions=n,
                 first_state_minute=360, first_target_minute=365, final_state_minute=records[-1]['minute'],
                 pump_changed_count=int(np.count_nonzero(np.abs(delivered-requested) > 1e-7)),
                 max_pump_difference_u_h=float(np.max(np.abs(delivered-requested))),
                 normalized_boundary_clip_count=len(changed),
                 normalized_boundary_clips=[dict(transition=int(i), target_minute=rows[i]['minute'],
                    delivered_u_h=float(delivered[i]), requested_u_h=float(requested[i]),
                    raw_normalized=float(raw_action[i]), stored_normalized=float(action[i])) for i in changed])
    return arrays, audit


def source_paths(config):
    return ([R/n for n in ('iql_wide.py', 'prepare_iql_replay.py', 'train_iql_wide.py',
                           'iql_wide_worker.py', 'physiologic_features.py', 'protocol.json')]
            + [HISTORY_SOURCE, SCORER, NORMALIZER_PATH, P/config['pump_source'], P/config['pump_parameters']]
            + [UPSTREAM/n for n in ('actor.py', 'critic.py', 'learner.py', 'policy.py',
                                    'configs/mujoco_config.py', 'upstream.json', 'LICENSE')])


def selected_episodes(config, data_hashes):
    """Read only declared train sources; never inspect development results."""
    natural = R/'results'/config['natural_name']
    manifest = read_json(natural/'manifest.json'); summary = read_json(natural/'summary.json')
    protocol = read_json(R/'protocol.json')
    if sha(SCORER) != protocol['scorer_sha256'] or str(SCORER.relative_to(P)) != protocol['scorer']:
        raise ValueError('Original scorer path/hash changed')
    if manifest['config']['split'] != 'train' or manifest['config']['smoke'] or summary['status'] != 'completed':
        raise ValueError('Natural data is not complete formal training collection')
    for key in ('train_scenario_seeds', 'patients', 'bolus_factors', 'warmup_minutes', 'total_minutes', 'scorer_sha256'):
        if manifest['protocol'][key] != protocol[key]:
            raise ValueError('Natural contract differs from frozen protocol: '+key)
    jobs = {'%s_p%02d_s%d' % (j['group'], j['patient'], j['seed']): j for j in manifest['jobs']}
    expected = {(p, s, f) for p in protocol['patients'] for s in protocol['train_scenario_seeds'] for f in protocol['bolus_factors']}
    training_seeds = set(protocol['train_scenario_seeds']) | set(range(103201, 103217))
    held_out_seeds = set().union(*(set(protocol[k+'_scenario_seeds'])
                                 for k in ('world_validation', 'development', 'confirmation')))
    if training_seeds & held_out_seeds:
        raise ValueError('Training selection overlaps held-out scenario seeds')
    cells = {(j['patient'], j['seed'], j['bolus_factor']) for j in jobs.values()}
    entries = summary['episodes']
    if cells != expected or len(jobs) != 120 or len(entries) != 120 or {e['key'] for e in entries} != set(jobs):
        raise ValueError('Expected all 120 distinct natural training cases')
    for path in (natural/'manifest.json', natural/'summary.json'):
        data_hashes[str(path.relative_to(P))] = sha(path)
    selected = []
    for entry in sorted(entries, key=lambda e: e['key']):
        path = natural/(entry['key']+'.json'); raw = read_json(path)
        if (raw['job'] != jobs[entry['key']] or raw['metrics'] != entry['metrics']
                or raw['failure_reason'] is not None or entry['failure_reason'] is not None):
            raise ValueError('Natural job/score/completion mismatch')
        selected.append((path, raw, 'natural', True))
    ppo = R/'results'/config['ppo_name']; pc = read_json(ppo/'config.json')
    for key in ('seed', 'gamma', 'reward_profile', 'low_risk_weight', 'reward_scale', 'terminal_penalty'):
        if pc[key] != config[key]:
            raise ValueError('PPO reward/training contract mismatch: '+key)
    if pc['patients'] != protocol['patients'] or pc['copies'] != 2 or pc['base_scenario_seed'] != 103201 or pc['total_minutes'] != 4320:
        raise ValueError('PPO train scenario contract changed')
    for path in (ppo/'config.json', ppo/'provenance.json'):
        data_hashes[str(path.relative_to(P))] = sha(path)
    provenance = read_json(ppo/'provenance.json')
    for filename, digest in provenance['source_sha256'].items():
        path = ppo/'source'/filename
        if sha(path) != digest:
            raise ValueError('PPO archived source hash mismatch: '+filename)
        data_hashes[str(path.relative_to(P))] = digest
    prefix = []
    with (ppo/'history.jsonl').open('rb') as stream:
        for iteration in range(1, 9):
            line = stream.readline(); update = json.loads(line)
            checkpoint = ppo/('policy_iter%02d.pt' % iteration)
            if update['iteration'] != iteration or sha(checkpoint) != update['checkpoint_sha256']:
                raise ValueError('First eight complete PPO updates/checkpoints are required')
            prefix.append(line); data_hashes[str(checkpoint.relative_to(P))] = sha(checkpoint)
    metadata = dict(ppo_history_first8_sha256=hashlib.sha256(b''.join(prefix)).hexdigest(),
                    ppo_history_first8_lines=[json.loads(line) for line in prefix])
    meals_by_seed = {}
    for iteration in range(8):
        for case in range(20):
            path = ppo/'trajectories'/('iter%02d_case%02d.json' % (iteration+1, case)); raw = read_json(path)
            patient = pc['patients'][case//2]; copy_index = case % 2
            seed = 103201+iteration*2+copy_index
            factor = [.8, 1., 1.2][(patient+copy_index+iteration) % 3]
            job = raw['job']
            if any(job[k] != v for k, v in dict(patient=patient, seed=seed, bolus_factor=factor,
                                               group='train_b%.1f' % factor, total_minutes=4320).items()):
                raise ValueError('PPO case identity differs from frozen first-eight schedule')
            if seed in meals_by_seed and meals_by_seed[seed] != job['meals']:
                raise ValueError('Meal realization differs between paired seed cases')
            meals_by_seed[seed] = job['meals']
            selected.append((path, raw, 'ppo_first8', False))
    if len(selected) != 280:
        raise ValueError('Expected exactly 280 training episodes')
    return selected, metadata


class Replay:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory/'manifest.json')
        if self.manifest['status'] != 'complete' or self.manifest['config'] != config:
            raise ValueError('Replay/config mismatch')
        for filename, digest in self.manifest['array_sha256'].items():
            if sha(self.directory/filename) != digest:
                raise ValueError('Replay array hash mismatch: '+filename)
        for filename, digest in self.manifest['source_sha256'].items():
            if sha(P/filename) != digest:
                raise ValueError('Replay feature/source contract changed: '+filename)
        if sha(self.directory/'episodes.json') != self.manifest['episodes_sha256']:
            raise ValueError('Replay episode metadata changed')
        self.arrays = {Path(n).stem: np.load(self.directory/n, mmap_mode='r', allow_pickle=False)
                       for n in self.manifest['array_sha256']}
        self.config = config; self.size = len(self.arrays['start'])

    def batch(self, indices):
        a = self.arrays; start = a['start'][indices]; anchor = a['anchor'][indices]
        result = {}
        for name, ends in [('state', start), ('next_state', start+1)]:
            result[name] = encode_observations(history_windows(a['frames'], ends), anchor,
                                               self.config, a['physiology'][ends])
        for name in ('action', 'reward', 'terminal'):
            result[name] = np.asarray(a[name][indices]).reshape(-1, 1)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); config = load_config(args.config)
    output = (R/config['replay']).resolve()
    try:
        output.relative_to(R/'data')
    except ValueError:
        raise ValueError('Replay must be inside the new research data directory')
    if output.exists():
        raise ValueError('Replay requires one new path under the research data directory')
    for path_key, hash_key in [('pump_source', 'pump_source_sha256'), ('pump_parameters', 'pump_parameters_sha256')]:
        if sha(P/config[path_key]) != config[hash_key]:
            raise ValueError('Original pump source/parameter hash mismatch')
    if read_json(UPSTREAM/'upstream.json')['commit'] != config['upstream_commit']:
        raise ValueError('Author IQL source commit changed')
    data_hashes = {}; selected, selection = selected_episodes(config, data_hashes)
    chunks = {}; episodes = []; frame_offset = transition_offset = 0
    for eid, (path, raw, role, full) in enumerate(selected):
        arrays, audit = episode_arrays(raw, config, require_complete=full)
        arrays['start'] += frame_offset
        arrays['episode_id'] = np.full(audit['transitions'], eid, dtype=np.int32)
        data_hashes[str(path.relative_to(P))] = sha(path)
        episodes.append(dict(episode_id=eid, source=role, raw_path=str(path.relative_to(P)),
                             raw_sha256=data_hashes[str(path.relative_to(P))], job=raw['job'],
                             frame_start=frame_offset, frame_end_exclusive=frame_offset+audit['records'],
                             transition_start=transition_offset,
                             transition_end_exclusive=transition_offset+audit['transitions'], **audit))
        for key, value in arrays.items():
            chunks.setdefault(key, []).append(value)
        frame_offset += audit['records']; transition_offset += audit['transitions']
    sources = source_paths(config)+[args.config]
    source_hashes = {str(path.relative_to(P)): sha(path) for path in sources}
    output.mkdir(parents=True)
    array_hashes = {}
    for name, parts in chunks.items():
        path = output/(name+'.npy'); np.save(path, np.concatenate(parts), allow_pickle=False)
        array_hashes[path.name] = sha(path)
    write_json(output/'episodes.json', episodes)
    write_json(output/'manifest.json', dict(
        schema=1, status='complete', config=config, config_sha256=sha(args.config),
        source_sha256=source_hashes, raw_data_sha256=data_hashes, array_sha256=array_hashes,
        episodes_sha256=sha(output/'episodes.json'), episodes=280, natural_episodes=120, ppo_episodes=160,
        transitions=transition_offset, feature_names=list(FEATURE_NAMES),
        feature_order='flatten(72x22), physiology28, warmup_anchor/5',
        true_terminal_episodes=sum(e['terminal'] for e in episodes),
        time_limit_episodes=sum(e['time_limit'] for e in episodes),
        normalized_boundary_clip_count=sum(e['normalized_boundary_clip_count'] for e in episodes),
        selection=selection, no_development_or_confirmation_data=True,
        identifiers_metadata_only=True, true_BG_reward_only=True,
        action_semantics='actual delivered basal/observed warmup anchor-1; recorded microscopic pump-boundary clips only',
        reward='-(high Kovatchev risk +2*low Kovatchev risk)/120; true terminal additionally -100',
        numpy_version=np.__version__))
    print(json.dumps(dict(status='complete', output=str(output), episodes=280,
                          transitions=transition_offset, manifest_sha256=sha(output/'manifest.json'))))


if __name__ == '__main__':
    main()
