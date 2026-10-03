"""Independent short CUDA integration checks; never save model parameters."""
import ast
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_shared import agent_class, Cycling, Replay, SharedReplay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_function(first, second, name):
    def body(path):
        return next(x for x in ast.parse(path.read_text()).body
                    if isinstance(x, ast.FunctionDef) and x.name == name)
    assert ast.dump(body(first), include_attributes=False) == ast.dump(body(second), include_attributes=False)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(260915)
    np.random.seed(260915)
    config_path = ROOT / 'configs/shared_methods.json'
    configs = json.loads(config_path.read_text())
    initial_config_sha = sha(config_path)
    original_eval = ROOT.parent/'RL_DSENet_2026-09-17/revision/evaluate_extended.py'
    same_function(original_eval, ROOT/'evaluate_final.py', 'scenario')
    same_function(original_eval, ROOT/'evaluate_final.py', 'environment_worker')
    loop = Replay('train')
    sim = SharedReplay()
    assert loop.size == 1653421 and sim.size == 392832
    # Sampling evidence: first full pass contains every origin exactly once,
    # even when take() requests cross the boundary and append a new permutation.
    cycling = Cycling(loop.size, 260915)
    first = cycling.take(loop.size-17)
    across = cycling.take(34)
    torch.testing.assert_close(torch.sort(torch.cat([first, across[:17]]))[0],
                               torch.arange(loop.size, device='cuda'), rtol=0, atol=0)
    assert cycling.passes == 1 and cycling.cursor == 17
    del first, across, cycling
    lo, so = Cycling(loop.size, 260915), Cycling(sim.size, 260916)
    keys = ('state', 'next_state', 'action', 'next_action', 'reward', 'q_valid',
            'bootstrap_valid', 'next_action_available')
    batches = []
    for _ in range(3):
        left, right = loop.batch(lo.take(128)), sim.batch(so.take(128))
        for key in keys:
            assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
            assert left[key].is_cuda and right[key].is_cuda
        batch = {k: torch.cat([left[k], right[k]], 0) for k in keys}
        assert batch['state'].shape == (256, 1584)
        assert all(torch.isfinite(batch[k]).all() for k in keys)
        batches.append(batch)
    source_paths = [ROOT/'train_shared.py', ROOT/'shared_worker.py', ROOT/'evaluate_final.py',
                    ROOT/'shared_data.py', ROOT.parent/'RL进阶对比_2026-09-15/rl_data.py',
                    ROOT.parent/'RL进阶对比_2026-09-15/rl_algorithms.py']
    hashes = {str(p.relative_to(ROOT.parent)): sha(p) for p in source_paths}
    result = dict(formal_training=False, checkpoint_files_written=False,
                  loop_origins=loop.size, simulation_origins=sim.size,
                  mix_per_step={'Loop':128, 'shared_simulation':128},
                  unchanged_scenario_and_environment_AST=True,
                  cycling_first_pass_contains_all_Loop_origins=True,
                  cuda_batch_keys_dtype_shape_finite=True,
                  shared_manifest_sha256=sha(ROOT/'shared_replay/manifest.json'),
                  shared_config_sha256=initial_config_sha, source_sha256=hashes,
                  methods={})
    for key, entry in configs.items():
        torch.manual_seed(260915)
        source = ROOT.parent / entry['checkpoint']
        initial_sha = sha(source)
        ck = torch.load(source, map_location='cpu')
        agent = agent_class(key)(entry['config']).cuda()
        agent.load_state_dict(ck['agent'])
        agent.optimizers()
        actor_before = {k:v.detach().clone() for k,v in agent.actor.state_dict().items()}
        # Exercise the actual JSON worker, including per-case seeded noise.
        worker = subprocess.Popen([sys.executable, str(ROOT/'shared_worker.py'),
                                   '--checkpoint', str(source)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            assert json.loads(worker.stdout.readline())['ready']
            indices = [0, 128, 255]
            histories = batches[0]['state'][indices].reshape(3,72,22)
            case_keys = ['check_Loop', 'check_S_A', 'check_S_B']
            seeds = [91701, 91701, 91702]
            rngs = {k:np.random.default_rng(s+20000) for k,s in zip(case_keys,seeds)}
            maximum = 0.
            for order in [[0,1,2], [2,0,1]]:
                local_keys = [case_keys[i] for i in order]
                payload = dict(history=histories[order].cpu().tolist(), case_keys=local_keys,
                               scenario_seeds=[seeds[i] for i in order])
                worker.stdin.write(json.dumps(payload)+'\n');worker.stdin.flush()
                response = json.loads(worker.stdout.readline())
                assert 'error' not in response, response
                z = torch.tensor([rngs[k].normal() for k in local_keys], device='cuda', dtype=torch.float32)[:,None]
                with torch.no_grad():
                    expected = (10*(agent.act(histories[order].flatten(1), z)+1)).flatten().cpu().numpy().clip(0,20)
                actual = np.array(response['actions_u_h'])
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
        finally:
            worker.stdin.close()
            worker.wait(timeout=30)
        steps = [1,2,3] if key != 'lom' else [1, entry['config']['gmm_pretrain_steps']+1,
                                            entry['config']['gmm_pretrain_steps']+2]
        diagnostics = []
        for batch, step in zip(batches, steps):
            metrics = agent.update(batch, step)
            assert all(np.isfinite(v) for v in metrics.values()), metrics
            diagnostics.append(dict(step=step, metrics=metrics))
        assert any(not torch.equal(v, actor_before[k]) for k,v in agent.actor.state_dict().items())
        with torch.no_grad():
            action = agent.act(batches[0]['state'], torch.zeros(256,1,device='cuda'))
        assert action.shape == (256,1) and torch.isfinite(action).all()
        assert (action >= -1).all() and (action <= 1).all()
        assert sha(source) == initial_sha
        algorithm_source = Path(sys.modules[agent_class(key).__module__].__file__)
        result['methods'][key] = dict(checkpoint_sha256=initial_sha, checkpoint_unchanged=True,
                                     algorithm_source_sha256=sha(algorithm_source),
                                     finite_short_updates=diagnostics, actor_parameters_changed=True,
                                     output_range_normalized=[float(action.min()),float(action.max())],
                                     worker_two_requests_permuted_case_order_max_error_u_h=maximum)
        print(json.dumps({'method':key,'checks':'passed','worker_max_error_u_h':maximum}), flush=True)
        del agent, ck, actor_before
        gc.collect();torch.cuda.empty_cache()
    assert sha(config_path) == initial_config_sha, 'Shared configs changed while checking'
    assert all(sha(ROOT.parent/p) == digest for p,digest in hashes.items()), 'Sources changed while checking'
    result['status'] = 'all_seven_real_mixed_CUDA_checks_passed'
    (ROOT/'checks/shared_train_integration.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status':result['status'],'methods':list(result['methods'])}), flush=True)


if __name__ == '__main__':
    main()
