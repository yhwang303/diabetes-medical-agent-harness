"""Public adapter checks: real paired arm dedup, boundaries, terminal and clocks."""
import argparse
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_shared_replay import Builder, require_complete_ppo, sha, status_score
from shared_data import SharedReplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-sample', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    builder = Builder()
    # Last feature is the action's successor. Distinct episodes cannot link.
    features = np.zeros((76, 20), dtype='float32')
    features[:, 0] = np.arange(76)
    action = np.array([.5, .75, 1., 1.25], dtype='float32')
    cgm = np.array([6., 5., 4., 3.], dtype='float32')
    bg = np.array([6.1, 5.1, 4.1, 3.1], dtype='float32')
    source = {'role': 'synthetic_public_check'}
    builder.add(features, action, cgm, bg, False, source)
    builder.add(features.copy(), action.copy(), cgm.copy(), bg.copy(), False, source)
    assert builder.duplicates == 1 and len(builder.episodes) == 1
    builder.add(features+100, action, cgm, bg, True, source)
    if args.paired_sample:
        with np.load(args.paired_sample, allow_pickle=False) as z:
            for arm in range(z['features'].shape[1]):
                n = int(z['length'][0, arm])
                builder.add(z['features'][0, arm, :72+n], z['action'][0, arm, :n],
                            z['target'][0, arm, :n], z['bg'][0, arm, :n],
                            z['terminal'][0, arm], {'role': 'real_paired_sample', 'arm': arm})
    arrays, episodes = builder.arrays()
    assert arrays['q_valid'][:8].tolist() == [True, True, True, False]*2
    assert arrays['terminal'][:8].tolist() == [False]*7+[True]
    assert arrays['time_limit'][:8].tolist() == [False]*3+[True]+[False]*4
    assert arrays['bootstrap_valid'][3] and not arrays['bootstrap_valid'][7]
    # Check local expression against the actual frozen pipeline function AST.
    source = (ROOT.parent/'Loop数据集/训练管线_v2/pipeline.py').read_text()
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'paper_status_score')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen_public_function>', 'exec'), scope)
    glucose = np.array([1., 2.9, 54/18, 69.999/18, 70/18, 120/18, 180/18, 360/18, 40.])
    np.testing.assert_array_equal(status_score(glucose), scope['paper_status_score'](glucose))
    with tempfile.TemporaryDirectory(prefix='shared_check_', dir=ROOT/'checks') as temp:
        path = Path(temp)
        try:
            require_complete_ppo(path)
        except (FileNotFoundError, RuntimeError):
            pass
        else:
            raise AssertionError('Incomplete official collection accepted')
        hashes = {}
        for key, value in arrays.items():
            np.save(path/(key+'.npy'), value)
            hashes[key+'.npy'] = sha(path/(key+'.npy'))
        (path/'episodes.json').write_text(json.dumps(episodes))
        manifest = dict(status='complete', schema_version=1, fixture_only=True,
                        origins=len(arrays['start']), array_sha256=hashes,
                        episodes_sha256=sha(path/'episodes.json'))
        (path/'manifest.json').write_text(json.dumps(manifest))
        data = SharedReplay(path, device='cpu')
        batch = data.batch(torch.tensor([0, 3, 4, 7]))
        assert batch['state'].shape == (4, 1584)
        assert torch.equal(batch['state'].reshape(4,72,22)[0,:,0], torch.arange(72).float())
        assert torch.equal(batch['next_state'].reshape(4,72,22)[1,:,0], torch.arange(4,76).float())
        assert batch['state'].reshape(4,72,22)[2,0,0] == 100
        db = data.ditr_batch([0, 3, 4, 7], horizon=12, rng=np.random.default_rng(260915))
        assert db['length'].tolist() == [4, 1, 4, 1]
        assert db['terminal'].tolist() == [False, False, True, True]
        assert torch.equal(db['action'][0,:4], torch.tensor(action))
        assert db['mask'].sum().item() == 10
        assert db['final_state'][:, -1, 0].tolist() == [75., 75., 175., 175.]
        short = data.ditr_batch([4], horizon=2)
        assert not short['terminal'].item(), 'Horizon ending before terminal cannot be terminal'
        # Hash verification actually rejects a modified array.
        with (path/'reward.npy').open('ab') as f:
            f.write(b'x')
        try:
            SharedReplay(path, device='cpu')
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed replay source accepted')
    result = dict(status='public_adapter_checks_passed', no_training=True,
                  checks=['trajectory_dedup', 'episode_boundaries', 'actual_action_units',
                          'state_clock', 'one_step_successor', 'reward_frozen_function_exact',
                          'true_terminal_vs_horizon_truncation', 'multistep_masks',
                          'incomplete_PPO_rejected', 'array_hash_tamper_rejected'],
                  real_paired_sample=str(args.paired_sample) if args.paired_sample else None,
                  real_paired_sample_sha256=sha(args.paired_sample) if args.paired_sample else None,
                  source_sha256={p.name:sha(p) for p in [ROOT/'prepare_shared_replay.py', ROOT/'shared_data.py', Path(__file__)]})
    (ROOT/'checks/shared_adapter_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
