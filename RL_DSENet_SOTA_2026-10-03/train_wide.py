"""Single-seed wide-action PPO using original real simulator transitions."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(P/'RL_DSENet_公平低糖_2026-09-21'))
from ppo_env import environment_worker,scenario


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    config_path=Path(args.config).resolve();c=json.loads(config_path.read_text())
    assert c['seed']==260915
    out=R/'results'/c['name'];out.mkdir(parents=True,exist_ok=False)
    (out/'trajectories').mkdir();(out/'source').mkdir()
    sources=[Path(__file__),R/'ppo_wide_worker.py',R/'physiologic_features.py',
             P/'RL_DSENet_公平低糖_2026-09-21/ppo_env.py']
    provenance=dict(config=c,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    reward_BG_not_inference_input=True,training_seed=260915)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for p in sources:(out/'source'/p.name).write_bytes(p.read_bytes())
    gpu=subprocess.Popen([str(P/'.venv-native/bin/python'),'-u',str(R/'ppo_wide_worker.py'),
                          '--config',str(config_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    ready=json.loads(gpu.stdout.readline());assert ready.get('ready'),ready
    context=mp.get_context('spawn');active=[];start=time.time()
    try:
        with (out/'history.jsonl').open('w',buffering=1) as log:
            for iteration in range(c['iterations']):
                active=[];outcomes={};anchors={}
                for patient in c['patients']:
                    for copy in range(c['copies']):
                        index=len(active);seed=c['base_scenario_seed']+iteration*c['copies']+copy
                        factor=[.8,1.,1.2][(patient+copy+iteration)%3]
                        job=dict(patient=patient,seed=seed,group='train_b%.1f'%factor,bolus_factor=factor,
                                 total_minutes=c['total_minutes'],meals=scenario(seed,c['total_minutes']))
                        parent,child=context.Pipe();proc=context.Process(target=environment_worker,args=(child,job))
                        proc.start();child.close();active.append((index,parent,proc))
                step=0
                while active:
                    pending=[];histories=[];ids=[]
                    for key,conn,proc in active:
                        if not conn.poll(180):raise TimeoutError('Training simulator did not respond')
                        message=conn.recv()
                        if message['done']:
                            proc.join(10);raw=message['result'];conn.close()
                            (out/'trajectories'/('iter%02d_case%02d.json'%(iteration+1,key))).write_text(json.dumps(raw,allow_nan=False))
                            if raw['failure_reason'] not in (None,'native_environment_done'):
                                raise RuntimeError('Technical simulator failure: '+str(raw['failure_reason']))
                            rows=[r for r in raw['records'] if not r['warmup']]
                            outcomes[str(key)]=dict(bg=[r['bg_mg_dl'] for r in rows],
                                                   terminal=raw['failure_reason']=='native_environment_done',
                                                   history=message['final_history'],anchor=anchors[key])
                        else:
                            h=np.asarray(message['history'],dtype=np.float32)
                            if key not in anchors:
                                assert h[-1,6]>.5
                                anchors[key]=float((h[-1,1]*.14462788945609448+.09945811581924525)*12)
                            ids.append(key);histories.append(h.tolist());pending.append((key,conn,proc))
                    active=pending
                    if not active:break
                    gpu.stdin.write(json.dumps(dict(op='act',indices=ids,history=histories,anchors=[anchors[k] for k in ids]))+'\n');gpu.stdin.flush()
                    reply=json.loads(gpu.stdout.readline())
                    assert len(reply['actions'])==len(active),reply
                    for (_,conn,_),action in zip(active,reply['actions']):conn.send(dict(action_u_h=action))
                    step+=1
                    if step%144==0:
                        print(json.dumps(dict(event='rollout',iteration=iteration+1,step=step,active=len(active),seconds=time.time()-start)),flush=True)
                gpu.stdin.write(json.dumps(dict(op='update',outcomes=outcomes))+'\n');gpu.stdin.flush()
                result=json.loads(gpu.stdout.readline());assert 'error' not in result,result
                result['wall_seconds']=time.time()-start;log.write(json.dumps(result)+'\n')
                (out/'progress.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(result),flush=True)
        (out/'completion.json').write_text(json.dumps(dict(status='completed',iterations=c['iterations'],wall_seconds=time.time()-start)))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),worker_returncode=gpu.poll())))
        raise
    finally:
        for _,conn,proc in active:
            if proc.is_alive():proc.terminate();proc.join()
        if gpu.poll() is None:
            gpu.stdin.write(json.dumps(dict(op='close'))+'\n');gpu.stdin.flush();gpu.wait(timeout=30)


if __name__=='__main__':main()
