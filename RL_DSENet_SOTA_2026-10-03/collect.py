"""Collect observable-history control episodes with the original simulator/scorer."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
sys.path.insert(0, str(P/'RL_DSENet_公平低糖_2026-09-21'))
from ppo_env import environment_worker, scenario


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_jobs(split):
    protocol = json.loads((R/'protocol.json').read_text())
    if split == 'confirmation':
        raise ValueError('Confirmation collection requires a separately frozen final runner')
    jobs = []
    for patient in protocol['patients']:
        for factor in protocol['bolus_factors']:
            for seed in protocol[split+'_scenario_seeds']:
                jobs.append(dict(patient=patient, seed=seed, group='bolus_%.1f'%factor,
                                 bolus_factor=factor, total_minutes=protocol['total_minutes'],
                                 meals=scenario(seed, protocol['total_minutes'])))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--split', choices=['train','world_validation','development'], required=True)
    ap.add_argument('--controller', choices=['explore','physiology','hold'], required=True)
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    from controller_baselines import make_controller
    jobs = make_jobs(args.split)
    if args.smoke:
        jobs = [jobs[0], next(j for j in jobs if j['patient']==9 and j['bolus_factor']==1.2)]
    out = R/'results'/args.name
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((R/'protocol.json').read_text())
    scorer = P/protocol['scorer']
    assert sha(scorer) == protocol['scorer_sha256']
    manifest = dict(config=vars(args), jobs=jobs, protocol=protocol,
                    source_sha256={p.name:sha(p) for p in
                                   [Path(__file__),R/'controller_baselines.py',R/'physiologic_features.py',
                                    P/'RL_DSENet_公平低糖_2026-09-21/ppo_env.py',scorer]})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    begin = time.time()
    context = mp.get_context('spawn')
    completed = []
    active = []
    try:
        for offset in range(0,len(jobs),args.batch_size):
            active = []
            for index,job in enumerate(jobs[offset:offset+args.batch_size],offset):
                parent,child = context.Pipe()
                process = context.Process(target=environment_worker,args=(child,job))
                process.start(); child.close()
                # Controller RNG is an experiment stream; patient ID is not supplied.
                controller = make_controller(args.controller, seed=260915+index)
                active.append(dict(pipe=parent,process=process,job=job,controller=controller,anchor=None))
            while active:
                pending = []
                for item in active:
                    conn = item['pipe']
                    if not conn.poll(120):
                        raise TimeoutError('Simulator worker stopped responding')
                    message = conn.recv()
                    if message['done']:
                        item['process'].join(10)
                        result = message['result'];job = result['job']
                        key = '%s_p%02d_s%d'%(job['group'],job['patient'],job['seed'])
                        (out/(key+'.json')).write_text(json.dumps(result,allow_nan=False))
                        completed.append(dict(key=key,patient=job['patient'],group=job['group'],seed=job['seed'],
                                              failure_reason=result['failure_reason'],metrics=result['metrics']))
                        conn.close()
                        print(json.dumps(dict(event='episode_complete',key=key,fail=result['failure_reason'],
                                              completed=len(completed),total=len(jobs),seconds=time.time()-begin)),flush=True)
                        (out/'progress.json').write_text(json.dumps(dict(completed=len(completed),total=len(jobs))))
                    else:
                        history = np.asarray(message['history'],dtype=np.float32)
                        if item['anchor'] is None:
                            assert history[-1,6]>.5
                            item['anchor'] = float((history[-1,1]*.14462788945609448+.09945811581924525)*12)
                        action = float(item['controller'].action(history[None],np.array([item['anchor']]))[0])
                        assert np.isfinite(action) and 0<=action<=min(20,2*item['anchor'])+1e-6
                        conn.send(dict(action_u_h=action));pending.append(item)
                active = pending
        (out/'summary.json').write_text(json.dumps(dict(status='completed',episodes=completed,
                                                       count=len(completed),wall_seconds=time.time()-begin),indent=2))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),completed=len(completed))))
        raise
    finally:
        for item in active:
            if item['process'].is_alive():item['process'].terminate();item['process'].join()


if __name__=='__main__':
    main()
