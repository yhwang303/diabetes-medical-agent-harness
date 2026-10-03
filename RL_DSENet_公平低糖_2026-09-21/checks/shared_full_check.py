"""Read-only full replay validation; writes only its separate check evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_shared_replay import sha, status_score
from shared_data import SharedReplay


def main():
    torch.set_num_threads(1)
    directory = ROOT / 'shared_replay'
    replay = SharedReplay(directory, device='cpu', verify=True)
    manifest = replay.manifest
    episodes = json.loads((directory / 'episodes.json').read_text())
    a = {k: v.numpy() for k, v in replay.arrays.items()}
    assert replay.size == 392832 and len(episodes) == 5704
    assert int(a['terminal'].sum()) == 0
    assert int(a['time_limit'].sum()) == len(episodes)
    assert len(a['features']) == replay.size + 72*len(episodes)
    np.testing.assert_array_equal(a['reward'], (status_score(a['cgm_mmol_l'])/12).astype('float32'))
    np.testing.assert_array_equal(a['bg_reward'], (status_score(a['bg_mg_dl']/18)/12).astype('float32'))
    sources_checked = 0
    for relative, expected in manifest['source_sha256'].items():
        assert sha(ROOT.parent / relative) == expected, relative
        sources_checked += 1
    ppo_raw_rows = 0
    for ep in episodes:
        start, n = ep['transition_offset'], ep['length']
        ids = np.arange(start, start+n)
        assert (a['episode_id'][ids] == ep['episode_id']).all()
        np.testing.assert_array_equal(a['episode_remaining'][ids], np.arange(n, 0, -1))
        np.testing.assert_array_equal(a['start'][ids], ep['feature_offset']+71+np.arange(n))
        assert not a['q_valid'][ids[-1]] and a['q_valid'][ids[:-1]].all()
        assert not a['next_action_available'][ids[-1]]
        np.testing.assert_array_equal(a['next_action'][ids[:-1]], a['action'][ids[1:]])
        ppo = [s for s in ep['sources'] if s['role'] == 'PPO_real_rewards']
        if ppo:
            raw = json.loads((ROOT.parent / ppo[0]['path']).read_text())
            controlled = [r for r in raw['records'] if not r['warmup']]
            raw_bg = np.array([r['bg_mg_dl'] for r in controlled])
            np.testing.assert_array_equal(a['bg_mg_dl'][ids], raw_bg)
            reward = (-((raw_bg-120)/60)**2 - 4*(np.maximum(70-raw_bg, 0)/16)**2).clip(-20, 0)/12
            np.testing.assert_array_equal(a['ppo_reward'][ids], reward.astype('float32'))
            ppo_raw_rows += n
    assert ppo_raw_rows == 126720
    # Include first/last positions of old and new data in the actual device API.
    selection = np.array([0, 47, 48, 266111, 266112, replay.size-1])
    batch = replay.batch(selection)
    assert batch['state'].shape == (6, 1584)
    assert all(torch.isfinite(batch[k]).all() for k in ['state', 'next_state', 'action', 'reward'])
    multi = replay.ditr_batch(selection, horizon=12, rng=np.random.default_rng(260915))
    assert not multi['terminal'].any()
    assert multi['length'].tolist() == [12, 1, 12, 1, 12, 1]
    assert torch.equal(multi['mask'].sum(1), multi['length'])
    result = dict(status='full_shared_replay_checks_passed', training_run=False,
                  manifest_sha256=sha(directory/'manifest.json'),
                  arrays_verified=len(manifest['array_sha256']), sources_verified=sources_checked,
                  episodes_verified=len(episodes), transitions_verified=replay.size,
                  ppo_original_BG_and_PPO_reward_exact_rows=ppo_raw_rows,
                  all_reward_rows_frozen_formula_exact=True,
                  no_cross_episode_successors=True, true_terminal_transitions=0,
                  real_CPU_batch_and_DITR_multistep=True,
                  source_sha256=sha(Path(__file__)))
    (ROOT/'checks/shared_full_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
