"""New-policy PPO on the unchanged public simulator, with frozen-world features.

This runner reads training protocol and completed-world metadata only. It never
loads development/confirmation trajectories or selects a policy checkpoint.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent
FIXED = dict(seed=260915, actor_lr=.0003, value_lr=.001, gamma=.997, gae_lambda=.99,
             epochs=4, batch_size=512, clip=.2, entropy=.01, gradient_clip=.5,
             teacher_kl=0., reward_profile='risk', terminal_penalty=100.,
             low_risk_weight=2., reward_scale=10., torch_threads=2,
             precision='float32_no_amp_no_tf32',
             action_multipliers=[0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config, protocol, bound=False):
    """Pure metadata gate, usable before NumPy/Torch or simulation imports."""
    c = config
    if any(c.get(k) != value for k, value in FIXED.items()):
        raise ValueError('Frozen PPO optimization/action/precision configuration changed')
    if protocol['training_seed'] != 260915 or c['world_variant'] not in ('point', 'quantile'):
        raise ValueError('Training seed or world variant mismatch')
    if Path(c['name']).name != c['name'] or c['name'] in ('', '.', '..'):
        raise ValueError('A new simple output directory name is required')
    if c['run_mode'] == 'formal':
        expected = dict(iterations=40, patients=protocol['patients'], copies=2,
                        base_scenario_seed=103201, total_minutes=protocol['total_minutes'],
                        allow_world_smoke_budget=False, development_checkpoints=[8, 16, 32, 40])
        low, high = protocol['ppo_train_scenario_seed_range_inclusive']
        allowed = set(range(low, high + 1))
    elif c['run_mode'] == 'smoke':
        expected = dict(iterations=1, patients=[1, 9], copies=1, base_scenario_seed=103800,
                        total_minutes=720, allow_world_smoke_budget=True, development_checkpoints=[])
        allowed = set(protocol['smoke_scenario_seeds'])
    else:
        raise ValueError('Expected explicit formal or smoke run mode')
    if any(c.get(k) != value for k, value in expected.items()):
        raise ValueError('Frozen formal/smoke population and training budget changed')
    seeds = set(range(c['base_scenario_seed'], c['base_scenario_seed'] + c['iterations'] * c['copies']))
    forbidden = set(protocol['development_scenario_seeds'] + protocol['confirmation_scenario_seeds']
                    + protocol['world_validation_scenario_seeds'])
    if not seeds <= allowed or seeds & forbidden:
        raise ValueError('Nontraining scenario seed entered the policy rollout schedule')
    if bound:
        for key in ('world_checkpoint_sha256', 'world_training_provenance_sha256', 'world_completion_sha256'):
            value = c.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
                raise ValueError('Missing resolved world SHA binding: ' + key)
        if not Path(c['world_checkpoint']).is_absolute() or not Path(c['output_dir']).is_absolute():
            raise ValueError('Resolved checkpoint/output paths must be absolute')
        Path(c['output_dir']).resolve().relative_to((R / 'results').resolve())
    return sorted(seeds)


def receive_worker(worker):
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError('Policy worker exited without a JSON reply; see worker_stderr.log')
    reply = json.loads(line)
    if 'error' in reply:
        raise RuntimeError('Policy worker failed: ' + str(reply['error']))
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', help='New output directory name; never overwrite an existing run')
    parser.add_argument('--world-checkpoint', help='Completed world checkpoint; relative paths are research-root relative')
    args = parser.parse_args()
    requested_path = Path(args.config).resolve()
    c = json.loads(requested_path.read_text())
    if args.name is not None: c['name'] = args.name
    if args.world_checkpoint is not None: c['world_checkpoint'] = args.world_checkpoint
    protocol = json.loads((R / 'protocol.json').read_text())
    seeds = validate_config(c, protocol)
    if sha(P / protocol['scorer']) != protocol['scorer_sha256']:
        raise ValueError('The original independent scorer changed')
    import numpy as np
    from world_control_worker import inspect_provenance
    checkpoint = Path(c['world_checkpoint'])
    if not checkpoint.is_absolute(): checkpoint = R / checkpoint
    checkpoint = checkpoint.resolve()
    provenance_path, world_provenance, world_sources, _ = inspect_provenance(checkpoint)
    if world_provenance['config']['variant'] != c['world_variant']:
        raise ValueError('Requested world variant and completed training run disagree')
    if world_provenance['config']['budget_override'] and not c['allow_world_smoke_budget']:
        raise ValueError('Formal PPO requires the full completed world training budget')
    out = R / 'results' / c['name']
    c.update(world_checkpoint=str(checkpoint), world_checkpoint_sha256=sha(checkpoint),
             world_training_provenance_sha256=sha(provenance_path),
             world_completion_sha256=sha(provenance_path.parent / 'completion.json'), output_dir=str(out))
    validate_config(c, protocol, bound=True)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('trajectories', 'source'): (out / name).mkdir()
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    shutil.copyfile(requested_path, out / 'source' / 'requested_config.json')
    sys.path.insert(0, str(P / 'RL_DSENet_公平低糖_2026-09-21'))
    from ppo_env import environment_worker, scenario
    import ppo_env, evaluate, observable_history, control_metrics, brake
    sources = set(world_sources + [Path(__file__).resolve(), R / 'ppo_world_worker.py', R / 'ppo_wide_worker.py',
                  R / 'physiologic_features.py', R / 'world_control_worker.py', R / 'protocol.json'])
    sources.update(Path(module.__file__).resolve() for module in
                   (ppo_env, evaluate, observable_history, control_metrics, brake))
    source_hashes = {}
    for path in sorted(sources):
        relative = path.relative_to(P); target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha(path)
    provenance = dict(config=c, requested_config_sha256=sha(requested_path), source_sha256=source_hashes,
                      protocol_sha256=sha(R / 'protocol.json'), training_seed=260915, scenario_seeds=seeds,
                      world_training_provenance=world_provenance, world_training_provenance_sha256=sha(provenance_path),
                      reward_BG_not_inference_input=True, independent_scorer_sha256=protocol['scorer_sha256'],
                      policy_training_data='new real simulator rollouts only', model_imagined_rollouts=False,
                      policy_checkpoint_selection='not performed by this runner',
                      known_virtual_patients=True, development_or_confirmation_loaded=False)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    context = mp.get_context('spawn'); active = []; processes = []; gpu = None
    start = time.time()
    try:
        with (out / 'worker_stderr.log').open('w') as errors:
            gpu = subprocess.Popen([str(P / '.venv-native/bin/python'), '-u', str(R / 'ppo_world_worker.py'),
                                    '--config', str(out / 'config.json')], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
            ready = receive_worker(gpu)
            if not ready.get('ready') or ready.get('feature_dim') != 195:
                raise RuntimeError('World policy initialization contract failed')
            (out / 'worker_ready.json').write_text(json.dumps(ready, indent=2))
            with (out / 'history.jsonl').open('w', buffering=1) as log:
                for iteration in range(c['iterations']):
                    active = []; processes = []; outcomes = {}; anchors = {}
                    for patient in c['patients']:
                        for copy in range(c['copies']):
                            index = len(active); seed = c['base_scenario_seed'] + iteration * c['copies'] + copy
                            factor = [.8, 1., 1.2][(patient + copy + iteration) % 3]
                            job = dict(patient=patient, seed=seed, group='train_b%.1f' % factor,
                                       bolus_factor=factor, total_minutes=c['total_minutes'],
                                       meals=scenario(seed, c['total_minutes']))
                            parent, child = context.Pipe()
                            proc = context.Process(target=environment_worker, args=(child, job))
                            proc.start(); child.close(); active.append((index, parent, proc)); processes.append((parent, proc))
                    step = 0
                    while active:
                        pending = []; histories = []; ids = []
                        for key, conn, proc in active:
                            if not conn.poll(180): raise TimeoutError('Training simulator did not respond')
                            message = conn.recv()
                            if message['done']:
                                proc.join(10); raw = message['result']; conn.close()
                                (out / 'trajectories' / ('iter%02d_case%02d.json' % (iteration + 1, key))).write_text(
                                    json.dumps(raw, allow_nan=False))
                                if proc.is_alive() or proc.exitcode != 0:
                                    raise RuntimeError('Simulator process failed to exit cleanly after its final result')
                                if raw['failure_reason'] not in (None, 'native_environment_done'):
                                    raise RuntimeError('Technical simulator failure: ' + str(raw['failure_reason']))
                                rows = [row for row in raw['records'] if not row['warmup']]
                                if key not in anchors or not rows:
                                    raise RuntimeError('Simulator terminated before a policy action; retained raw failure')
                                outcomes[str(key)] = dict(bg=[row['bg_mg_dl'] for row in rows],
                                    terminal=raw['failure_reason'] == 'native_environment_done',
                                    failure_reason=raw['failure_reason'], history=message['final_history'], anchor=anchors[key])
                            else:
                                history = np.asarray(message['history'], dtype=np.float32)
                                if key not in anchors:
                                    if history[-1, 6] <= .5: raise ValueError('Warmup basal must be observed')
                                    anchors[key] = float((history[-1, 1] * .14462788945609448 + .09945811581924525) * 12)
                                ids.append(key); histories.append(history.tolist()); pending.append((key, conn, proc))
                        active = pending
                        if not active: break
                        request = dict(op='act', indices=ids, history=histories, anchors=[anchors[key] for key in ids])
                        gpu.stdin.write(json.dumps(request, allow_nan=False) + '\n'); gpu.stdin.flush()
                        reply = receive_worker(gpu)
                        if len(reply['actions']) != len(active): raise ValueError('Policy action batch mismatch')
                        for (_, conn, _), action in zip(active, reply['actions']): conn.send(dict(action_u_h=action))
                        step += 1
                        if step % 144 == 0:
                            print(json.dumps(dict(event='rollout', iteration=iteration + 1, step=step,
                                                  active=len(active), seconds=time.time() - start)), flush=True)
                    gpu.stdin.write(json.dumps(dict(op='update', outcomes=outcomes), allow_nan=False) + '\n'); gpu.stdin.flush()
                    result = receive_worker(gpu); result['wall_seconds'] = time.time() - start
                    log.write(json.dumps(result, allow_nan=False) + '\n')
                    (out / 'progress.json').write_text(json.dumps(result, indent=2, allow_nan=False))
                    print(json.dumps(result, allow_nan=False), flush=True)
            (out / 'completion.json').write_text(json.dumps(dict(status='completed_candidate_only',
                iterations=c['iterations'], run_mode=c['run_mode'], wall_seconds=time.time() - start,
                world_checkpoint_sha256=c['world_checkpoint_sha256'], last_checkpoint=result['checkpoint'],
                last_checkpoint_sha256=result['checkpoint_sha256'], checkpoint_selected=False)))
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(error),
            worker_returncode=None if gpu is None else gpu.poll(), wall_seconds=time.time() - start)))
        raise
    finally:
        for conn, proc in processes:
            conn.close()
            if proc.is_alive(): proc.terminate()
            proc.join(timeout=10)
        if gpu is not None and gpu.poll() is None:
            try:
                gpu.stdin.write(json.dumps(dict(op='close')) + '\n'); gpu.stdin.flush(); gpu.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                gpu.kill(); gpu.wait(timeout=10)


if __name__ == '__main__':
    main()
