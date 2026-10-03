"""Freeze public simulation records for all baselines, without changing Loop.

The official build refuses incomplete PPO collection. No training occurs here.
Only observed CGM, delivered insulin and recorded meals enter features; BG is a
label. Source roles, duplicates and all source hashes remain auditable.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
GAMMA = .9 ** (1 / 12)
NORMALIZER_SHA = '90ef9f4faf383d46102896a980f9ddd2220e302983d246cbc74aa88dd1ca4a5f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status_score(glucose_mmol_l):
    # Same frozen pipeline.paper_status_score; isolated from pandas/data loaders.
    g = np.asarray(glucose_mmol_l, dtype=np.float64)
    if not np.all(np.isfinite(g) & (g > 0)):
        raise ValueError('Reward requires observed positive glucose')
    b = g * 18.
    risk = 10 * (1.509 * (np.log(b) ** 1.084 - 5.381)) ** 2
    return np.where(b < 70., -1., 1. - np.clip(risk, 0, 15.5) / 7.75)


def trajectory_hash(features, action, cgm, bg, terminal):
    h = hashlib.sha256()
    for value in (features, action, cgm, bg):
        value = np.asarray(value, dtype='<f4', order='C')
        h.update(json.dumps(list(value.shape)).encode())
        h.update(value.tobytes())
    h.update(bytes([bool(terminal)]))
    return h.hexdigest()


class Builder:
    def __init__(self):
        self.episodes = []
        self.keys = {}
        self.sources = {}
        self.duplicates = 0

    def add(self, features, action, cgm, bg, terminal, source, raw_bg_mg_dl=None):
        features, action, cgm, bg = [np.asarray(v, dtype='float32')
                                      for v in (features, action, cgm, bg)]
        n = len(action)
        assert n > 0 and features.shape == (72 + n, 20)
        assert cgm.shape == bg.shape == (n,)
        assert all(np.isfinite(v).all() for v in (features, action, cgm, bg))
        assert ((action >= 0) & (action <= 20)).all()
        assert (cgm > 0).all() and (bg > 0).all()
        key = trajectory_hash(features, action, cgm, bg, terminal)
        if key in self.keys:
            self.episodes[self.keys[key]]['sources'].append(source)
            self.duplicates += 1
            return
        self.keys[key] = len(self.episodes)
        raw_bg = bg.astype('float64') * 18. if raw_bg_mg_dl is None else np.asarray(raw_bg_mg_dl, dtype='float64')
        assert raw_bg.shape == (n,) and np.isfinite(raw_bg).all() and (raw_bg > 0).all()
        self.episodes.append(dict(features=features, action=action, cgm=cgm,
                                  bg=bg, terminal=bool(terminal), hash=key,
                                  raw_bg=raw_bg, sources=[source]))

    def arrays(self):
        chunks = []
        metadata = []
        feature_offset = transition_offset = 0
        for eid, ep in enumerate(self.episodes):
            n = len(ep['action'])
            terminal = np.zeros(n, dtype=bool)
            terminal[-1] = ep['terminal']
            next_available = np.arange(n) < n - 1
            action = ep['action'] / 10. - 1.
            next_action = np.zeros(n, dtype='float32')
            next_action[:-1] = action[1:]
            bg_mg_dl = ep['raw_bg']
            ppo_reward = (-((bg_mg_dl.astype('float64') - 120.) / 60.) ** 2
                          - 4. * (np.maximum(70. - bg_mg_dl, 0.) / 16.) ** 2)
            ppo_reward = ppo_reward.clip(-20., 0.) / 12.
            ppo_reward[terminal] -= 20. / 12.
            chunks.append(dict(
                features=ep['features'],
                start=np.arange(n, dtype='int64') + feature_offset + 71,
                action=action.astype('float32'), next_action=next_action,
                action_u_h=ep['action'], cgm_mmol_l=ep['cgm'], bg_mg_dl=bg_mg_dl,
                reward=(status_score(ep['cgm']) / 12.).astype('float32'),
                bg_reward=(status_score(bg_mg_dl/18.) / 12.).astype('float32'),
                ppo_reward=ppo_reward.astype('float32'),
                q_valid=next_available & ~terminal,
                bootstrap_valid=~terminal,
                next_action_available=next_available,
                terminal=terminal,
                time_limit=(np.arange(n) == n - 1) & ~terminal,
                episode_remaining=np.arange(n, 0, -1, dtype='int32'),
                episode_id=np.full(n, eid, dtype='int32'),
                episode_step=np.arange(n, dtype='int32'),
            ))
            metadata.append(dict(episode_id=eid, feature_offset=feature_offset,
                                 transition_offset=transition_offset, length=n,
                                 terminal=ep['terminal'], trajectory_sha256=ep['hash'],
                                 sources=ep['sources']))
            feature_offset += n + 72
            transition_offset += n
        assert chunks
        arrays = {key: np.concatenate([x[key] for x in chunks]) for key in chunks[0]}
        return arrays, metadata


def require_complete_ppo(project):
    task = project / ROOT.name
    requested = json.loads((task / 'configs/ppo.json').read_text())
    assert requested['name'] == 'PPO_real_rewards' and requested['iterations'] == 8
    directory = task / 'results' / requested['name']
    path = directory / 'completion.json'
    if not path.exists():
        raise RuntimeError('Refusing shared replay: official 8-iteration PPO is not complete')
    complete = json.loads(path.read_text())
    assert complete['status'] == 'completed' and complete['iterations'] == 8
    config = json.loads((directory / 'config.json').read_text())
    assert config == requested, 'PPO run/config mismatch'
    assert config['seed'] == 260915 and config['patients'] == list(range(1, 11))
    assert config['copies'] == 2 and config['total_minutes'] == 4320
    expected = {f'iter{i:02d}_case{k:02d}.json'
                for i in range(1, 9) for k in range(20)}
    found = {p.name for p in (directory / 'trajectories').glob('*.json')}
    assert found == expected, 'PPO trajectory collection incomplete or contains extras'
    updates = [json.loads(s) for s in (directory / 'history.jsonl').read_text().splitlines()]
    assert [x['iteration'] for x in updates] == list(range(1, 9))
    for i, update in enumerate(updates, 1):
        assert sha(directory / f'policy_iter{i:02d}.pt') == update['checkpoint_sha256']
    return directory, config, updates


def add_paired(builder, project):
    for folder, arms in [('paired_sim_train', 5), ('paired_sim_train_temporal', 7)]:
        directory = project / 'RL_DITR创新_2026-09-16' / folder
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['contract']['scenario_seeds'] == [40101, 40102]
        assert manifest['contract']['patients'] == [1, 2, 3, 4]
        for fixed in ['manifest.json', 'contract.json']:
            builder.sources[str((directory / fixed).relative_to(project))] = sha(directory / fixed)
        count = intervals = 0
        for record in manifest['scenarios']:
            path = directory / record['file']
            assert sha(path) == record['sha256'], str(path)
            builder.sources[str(path.relative_to(project))] = record['sha256']
            with np.load(path, allow_pickle=False) as z:
                assert z['features'].shape == (record['groups'], arms, 120, 20)
                for group in range(record['groups']):
                    for arm in range(arms):
                        n = int(z['length'][group, arm])
                        assert np.array_equal(z['mask'][group, arm], np.arange(48) < n)
                        assert 0 < n <= 48
                        source = dict(role=folder, path=str(path.relative_to(project)),
                                      group=group, arm=arm, minute=int(z['minute'][group]),
                                      patient=record['patient'], scenario_seed=record['scenario_seed'],
                                      bolus_factor=record['bolus_factor'], basal_factor=record['basal_factor'])
                        builder.add(z['features'][group, arm, :72+n],
                                    z['action'][group, arm, :n], z['target'][group, arm, :n],
                                    z['bg'][group, arm, :n], z['terminal'][group, arm], source)
                        count += 1
                        intervals += n
        assert count == manifest['arms'] and intervals == manifest['valid_intervals']


def add_ppo(builder, project, directory, config, updates):
    path = project / 'RL进阶对比_2026-09-15/observable_history.py'
    spec = importlib.util.spec_from_file_location('shared_observable_history', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder.sources[str(path.relative_to(project))] = sha(path)
    for fixed in ['completion.json', 'config.json', 'history.jsonl', 'provenance.json']:
        builder.sources[str((directory / fixed).relative_to(project))] = sha(directory / fixed)
    transition_count = 0
    for iteration in range(1, 9):
        checkpoint = directory / f'policy_iter{iteration:02d}.pt'
        builder.sources[str(checkpoint.relative_to(project))] = updates[iteration-1]['checkpoint_sha256']
        for case in range(20):
            path = directory / 'trajectories' / f'iter{iteration:02d}_case{case:02d}.json'
            raw = json.loads(path.read_text())
            builder.sources[str(path.relative_to(project))] = sha(path)
            job = raw['job']
            patient = config['patients'][case // config['copies']]
            copy = case % config['copies']
            assert job['patient'] == patient
            assert job['seed'] == config['base_scenario_seed'] + (iteration-1)*config['copies'] + copy
            assert job['total_minutes'] == config['total_minutes']
            assert raw['failure_reason'] in [None, 'native_environment_done'], 'Failed infrastructure is not terminal evidence'
            records = raw['records']
            assert [x['minute'] for x in records] == list(range(5, len(records)*5+1, 5))
            assert len(records) > 72
            assert all(r['warmup'] == (i < 72) for i, r in enumerate(records))
            if raw['failure_reason'] is None:
                assert len(records)*5 == config['total_minutes']
            history = module.History()
            for row in records:
                assert all(np.isfinite(row[k]) for k in ['cgm_mg_dl', 'bg_mg_dl', 'delivered_basal_u_h', 'bolus_u', 'meal_g'])
                history.append(row['minute'], row['cgm_mg_dl'], row['delivered_basal_u_h']/12.,
                               row['bolus_u'] if row['bolus_u'] > 0 else None,
                               row['meal_g'] if row['meal_g'] > 0 else None)
            controlled = records[72:]
            source = dict(role='PPO_real_rewards', path=str(path.relative_to(project)),
                          iteration=iteration, case=case, patient=patient,
                          scenario_seed=job['seed'], bolus_factor=job['bolus_factor'])
            builder.add(np.array(history.rows), [x['delivered_basal_u_h'] for x in controlled],
                        [x['cgm_mg_dl']/18. for x in controlled],
                        [x['bg_mg_dl']/18. for x in controlled],
                        raw['failure_reason'] == 'native_environment_done', source,
                        raw_bg_mg_dl=[x['bg_mg_dl'] for x in controlled])
            transition_count += len(controlled)
        assert transition_count == updates[iteration-1]['transitions'], 'PPO logged transition count mismatch'


def build(project, destination):
    # Check collection before opening source data or creating an output directory.
    ppo_directory, ppo_config, updates = require_complete_ppo(project)
    if destination.exists():
        raise FileExistsError('Shared replay is immutable; use a new output path')
    normalizer = project / 'Loop数据集/训练管线_v2/prepared/normalization.json'
    assert sha(normalizer) == NORMALIZER_SHA
    builder = Builder()
    builder.sources[str(normalizer.relative_to(project))] = sha(normalizer)
    add_paired(builder, project)
    paired_count, paired_duplicates = len(builder.episodes), builder.duplicates
    add_ppo(builder, project, ppo_directory, ppo_config, updates)
    arrays, episodes = builder.arrays()
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for key, value in arrays.items():
        path = destination / (key + '.npy')
        np.save(path, value, allow_pickle=False)
        hashes[path.name] = sha(path)
    (destination / 'episodes.json').write_text(json.dumps(episodes, indent=2, allow_nan=False))
    manifest = dict(
        status='complete', schema_version=1, origin='public_simulator_only',
        loop_data_modified=False, loop_origins_to_retain=1653421,
        origins=len(arrays['start']), episodes=len(episodes), q_valid=int(arrays['q_valid'].sum()),
        paired_unique_episodes=paired_count, paired_duplicate_episodes=paired_duplicates,
        ppo_episodes=160, ppo_transitions=updates[-1]['transitions'],
        true_terminal_transitions=int(arrays['terminal'].sum()),
        truncations=int(arrays['time_limit'].sum()), gamma=GAMMA,
        reward='frozen paper_status_score(next observed CGM mmol/L)/12',
        bg_reward='same frozen status score on true simulator BG, label only',
        bg_mg_dl='float64; original PPO JSON BG preserved, old paired float32 mmol/L converted exactly by float64 times18',
        ppo_reward='new PPO quadratic BG risk, [-20,0]/12, true terminal adds -20/12; gamma=.997 in PPO only',
        q_valid_contract='observed next action within same trajectory; last steps retained for behavior, omitted from common TD',
        bootstrap_valid_contract='false only on genuine simulator terminal; a recorded path ending is not physiological terminal',
        state_shape=[72, 22], policy_features='20 observable normalized features + original 2 clock columns; no BG or patient ID',
        deduplication='SHA256 of complete float32 features/actions/CGM/BG path and terminal; all duplicate provenance retained',
        action_mapping='normalized=actual delivered U/h / 10 - 1; action_u_h preserves physical label',
        data_access_limit='matched raw simulation records and labels; does not equal identical objectives, on-policy access, optimization or model selection',
        source_sha256=builder.sources, array_sha256=hashes,
        episodes_sha256=sha(destination / 'episodes.json'),
        builder_sha256=sha(Path(__file__)), normalizer_sha256=NORMALIZER_SHA,
    )
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / ROOT.name / 'shared_replay'
    result = build(args.project_root.resolve(), output.resolve())
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('sha256')}, indent=2))
