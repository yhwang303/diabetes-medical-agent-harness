"""Common observed-data adaptation; algorithm losses and Loop origins retained."""
import argparse,hashlib,importlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;P=R.parent;OLD=P/'RL进阶对比_2026-09-15';REV=P/'RL_DSENet_2026-09-17/revision'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(REV))
from rl_data import Replay
from shared_data import SharedReplay

def agent_class(key):
 return importlib.import_module({'bc':'rl_algorithms','fql':'fql_corrected','rebrac':'rebrac_corrected'}.get(key,key+'_algorithm' if key!='lom' else 'lom_corrected')).Agent

class Cycling:
 def __init__(self,size,seed):self.size=size;self.g=torch.Generator(device='cuda').manual_seed(seed);self.ids=torch.randperm(size,device='cuda',generator=self.g);self.cursor=0;self.passes=0
 def take(self,n):
  out=[]
  while n:
   k=min(n,self.size-self.cursor);out.append(self.ids[self.cursor:self.cursor+k]);self.cursor+=k;n-=k
   if self.cursor==self.size:self.ids=torch.randperm(self.size,device='cuda',generator=self.g);self.cursor=0;self.passes+=1
  return torch.cat(out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);a=ap.parse_args();configs=json.loads((R/'configs/shared_methods.json').read_text());entry=configs[a.method]
 c=entry['config'];out=R/'results'/('shared_'+a.method);out.mkdir(exist_ok=False)
 torch.set_num_threads(2);torch.manual_seed(260915);np.random.seed(260915);torch.backends.cuda.matmul.allow_tf32=True
 source=P/entry['checkpoint'];ck=torch.load(source,map_location='cpu');agent=agent_class(a.method)(c).cuda();agent.load_state_dict(ck['agent']);agent.optimizers()
 loop=Replay('train');sim=SharedReplay();assert loop.size==1653421
 manifest_sha=hashlib.sha256((R/'shared_replay/manifest.json').read_bytes()).hexdigest();assert manifest_sha==entry['shared_manifest_sha256'], 'shared data freeze mismatch'
 lo=Cycling(loop.size,260915);so=Cycling(sim.size,260916)
 (out/'config.json').write_text(json.dumps(entry,indent=2));(out/'sources').mkdir()
 sources=[Path(__file__).resolve(),R/'shared_data.py',OLD/'rl_data.py',Path(sys.modules[agent_class(a.method).__module__].__file__).resolve(),OLD/'rl_algorithms.py',REV/'lom_algorithm.py']
 sources=list(dict.fromkeys(sources))
 for p in sources:shutil.copy2(p,out/'sources'/p.name)
 provenance=dict(initial_checkpoint=entry['checkpoint'],initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),loop_manifest=loop.manifest,shared_manifest=sim.manifest,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},seed=260915,selection='fixed final budget; no simulator-based checkpoint selection',same_raw_training_sources=True,identical_training_objectives_to_ours=False,loop_origins_retained=True)
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 start=time.time();logs=[];seen_l=0;seen_s=0
 try:
  with (out/'history.jsonl').open('w',buffering=1) as f:
   for step in range(1,c['updates']+1):
    lb=loop.batch(lo.take(128));sb=sim.batch(so.take(128));keys=('state','next_state','action','next_action','reward','q_valid','bootstrap_valid','next_action_available');batch={k:torch.cat([lb[k],sb[k]],dim=0) for k in keys}
    metrics=agent.update(batch,step);assert all(np.isfinite(v) for v in metrics.values());seen_l+=128;seen_s+=128;logs.append(metrics)
    if step%500==0:
     keys=set().union(*(z.keys() for z in logs));record={k:float(np.mean([z[k] for z in logs if k in z])) for k in keys};record.update(step=step,seconds=time.time()-start,loop_seen=seen_l,sim_seen=seen_s);logs=[];f.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
    if step%10000==0 or step==c['updates']:
     state=dict(agent=agent.state_dict(),config=c,step=step,optimizers={k:v.state_dict() for k,v in vars(agent).items() if isinstance(v,torch.optim.Optimizer)},rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),loop_seen=seen_l,sim_seen=seen_s)
     torch.save(state,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
  result=dict(status='completed',steps=c['updates'],loop_seen=seen_l,sim_seen=seen_s,loop_full_passes=lo.passes,sim_full_passes=so.passes,seconds=time.time()-start,checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest())
  assert hashlib.sha256(source.read_bytes()).hexdigest()==provenance['initial_sha256']
  assert hashlib.sha256((R/'shared_replay/manifest.json').read_bytes()).hexdigest()==manifest_sha
  for p in sources:assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['source_sha256'][p.name]
  (out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (out/'failure.json').write_text(json.dumps(dict(step=step,error=repr(e)),indent=2));raise
if __name__=='__main__':main()
