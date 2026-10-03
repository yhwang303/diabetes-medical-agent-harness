"""Public simulator PPO data collection, fixed single training seed and budget."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,multiprocessing as mp,subprocess,time
from pathlib import Path
import numpy as np
from ppo_env import environment_worker,scenario
R=Path(__file__).resolve().parent;P=R.parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 config='ppo_smoke' if a.smoke else 'ppo';c=json.loads((R/'configs'/(config+'.json')).read_text())
 out=R/'results'/c['name'];out.mkdir(exist_ok=False);(out/'trajectories').mkdir()
 (out/'config.json').write_text(json.dumps(c,indent=2));(out/'source').mkdir()
 for p in [R/'train_ppo.py',R/'ppo_env.py',R/'ppo_worker.py']:(out/'source'/p.name).write_bytes(p.read_bytes())
 (out/'provenance.json').write_text(json.dumps({'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').glob('*.py')},'frozen_weight_hashes':json.loads((R/'configs/retained_weights.json').read_text()),'actor_initialization':'D06 frozen checkpoint','actual_simulator_reward':True,'persistent_observed_anchor_in_training':True,'hidden_BG_only_training_reward_not_policy_feature':True,'single_training_seed':c['seed']},indent=2))
 gpu=subprocess.Popen([str(P/'.venv-native/bin/python'),str(R/'ppo_worker.py'),'--config',config],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 assert json.loads(gpu.stdout.readline())['ready'];ctx=mp.get_context('spawn');log=(out/'history.jsonl').open('w',buffering=1);start=time.time();active=[]
 try:
  for iteration in range(c['iterations']):
   active=[];outcomes={};states={};anchors={};jobs={}
   for p in c['patients']:
    for copy in range(c['copies']):
     key=len(jobs);seed=c['base_scenario_seed']+iteration*c['copies']+copy;factor=[.8,1.,1.2][(p+copy+iteration)%3]
     job=dict(patient=p,seed=seed,group='ppo_b%.1f'%factor,bolus_factor=factor,total_minutes=c['total_minutes'],meals=scenario(seed,c['total_minutes']))
     parent,child=ctx.Pipe();proc=ctx.Process(target=environment_worker,args=(child,job));proc.start();child.close();active.append((key,parent,proc));jobs[key]=job
   step=0
   while active:
    pending=[];current=[];ids=[]
    for key,conn,proc in active:
     if not conn.poll(180):raise TimeoutError('Training simulator stalled')
     message=conn.recv()
     if message['done']:
      proc.join(10);raw=message['result'];path=out/'trajectories'/('iter%02d_case%02d.json'%(iteration+1,key));path.write_text(json.dumps(raw,allow_nan=False));rows=[r for r in raw['records'] if not r['warmup']]
      outcomes[str(key)]=dict(bg=[r['bg_mg_dl'] for r in rows],terminal=raw['failure_reason'] is not None,history=message['final_history'],anchor=anchors[key]);conn.close()
     else:
      h=message['history']
      if key not in anchors:anchors[key]=float((h[-1,1]*.14462788945609448+.09945811581924525)*12)
      ids.append(key);current.append(h.tolist());pending.append((key,conn,proc))
    active=pending
    if not active:break
    gpu.stdin.write(json.dumps(dict(op='act',indices=ids,history=current,anchors=[anchors[k] for k in ids]))+'\n');gpu.stdin.flush();reply=json.loads(gpu.stdout.readline())
    assert len(reply['actions'])==len(active)
    for (_,conn,_),action in zip(active,reply['actions']):conn.send({'action_u_h':action})
    step+=1
    if step%144==0:print(json.dumps(dict(event='rollout',iteration=iteration+1,step=step,active=len(active),seconds=time.time()-start)),flush=True)
   gpu.stdin.write(json.dumps(dict(op='update',outcomes=outcomes))+'\n');gpu.stdin.flush();update=json.loads(gpu.stdout.readline());update['seconds']=time.time()-start;log.write(json.dumps(update)+'\n');print(json.dumps(update),flush=True)
  (out/'completion.json').write_text(json.dumps(dict(status='completed',iterations=c['iterations'],wall_seconds=time.time()-start,training_seed=c['seed']),indent=2))
 finally:
  if gpu.poll() is None:
   gpu.stdin.write(json.dumps({'op':'close'})+'\n');gpu.stdin.flush();gpu.wait(timeout=30)
  for _,conn,proc in active:
   if proc.is_alive():proc.terminate();proc.join()
if __name__=='__main__':main()
