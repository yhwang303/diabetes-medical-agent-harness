"""Tracked offline policy training. Offline diagnostics never stand in for policy evaluation."""
from pathlib import Path
import argparse,json,time,hashlib,shutil,traceback
import numpy as np
import torch
from rl_data import Replay,ROOT
from rl_algorithms import Agent

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

@torch.no_grad()
def diagnostic(agent,data,seed=91526,n=8192):
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(seed);ix=torch.randperm(data.size,device=data.device)[:n];actions=[];errors=[];qs=[];losses=[];counts=[]
  for i in range(0,len(ix),256):
   b=data.batch(ix[i:i+256]);a=agent.act(b['state']);actions.extend((10*(a+1)).flatten().tolist());errors.extend((10*(a-b['action'])).flatten().tolist())
   if agent.algo!='bc':
    loss,q,t=agent.critic_loss(b);m=b['q_valid'].flatten();qs.extend(q.mean(-1)[m].tolist());losses.append(float(loss));counts.append(int(m.sum()))
  errors=np.array(errors);actions=np.array(actions)
  return {'split':'validation','samples':len(ix),'action_mae_u_h':float(np.abs(errors).mean()),'action_rmse_u_h':float(np.sqrt(np.mean(errors**2))),'action_quantiles_u_h':np.quantile(actions,[0,.01,.1,.5,.9,.99,1]).tolist(),'action_saturation_fraction':float(((actions<.01)|(actions>19.99)).mean()),'td_loss':float(np.average(losses,weights=counts)) if losses else None,'q_mean':float(np.mean(qs)) if qs else None,'scope':'logged-action/TD diagnostics only; no policy effectiveness conclusion','clinical_ready':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);arg=ap.parse_args();cfg=json.loads(Path(arg.config).read_text());out=ROOT/'results'/cfg['name'];out.mkdir(exist_ok=True,parents=True)
 if (out/'provenance.json').exists():raise RuntimeError('Run already exists; use a new name and explicit resume if needed')
 write(out/'config.json',cfg);(out/'sources').mkdir(exist_ok=True)
 for p in ROOT.glob('*.py'):shutil.copy2(p,out/'sources'/p.name)
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
 DataClass=Replay
 if cfg.get('anchor_context'):
  from anchor_data import AnchorReplay
  DataClass=AnchorReplay
 train=DataClass('train');val=DataClass('validation');agent=Agent(cfg).cuda();agent.optimizers()
 guidance_provenance=None
 if cfg.get('model_guidance'):
  from model_guidance import ModelGuidance
  agent.model_guidance=ModelGuidance(cfg['model_guidance']);guidance_provenance=agent.model_guidance.provenance
 optimizers={k:getattr(agent,k) for k in ['actor_opt','q_opt','flow_opt'] if hasattr(agent,k)}
 step=0;seen=0;td_seen=0;full_passes=0;cursor=0;permutation=torch.randperm(train.size,device='cuda');parent=None
 if cfg.get('resume') or cfg.get('warm_start'):
  parent_name=cfg.get('resume') or cfg['warm_start'];path=ROOT/'results'/parent_name/'last.pt';ck=torch.load(path,map_location='cuda')
  allowed={'name','updates','resume','log_every','save_every'}
  if cfg.get('warm_start'):allowed|={'warm_start','model_guidance'}
  changes={k:(ck['config'].get(k),cfg.get(k)) for k in set(ck['config'])|set(cfg) if k not in allowed and ck['config'].get(k)!=cfg.get(k)}
  if changes:raise ValueError('Resume changes training semantics: '+str(changes))
  agent.load_state_dict(ck['agent'])
  for k,v in optimizers.items():v.load_state_dict(ck['optimizers'][k])
  step=ck['step'];seen=ck['seen'];td_seen=ck['td_seen'];full_passes=ck['full_passes'];cursor=ck['cursor'];permutation=ck['permutation'].cuda()
  torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng_cuda']]);parent={'run':parent_name,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kind':'explicit_model_guidance_objective_change' if cfg.get('warm_start') else 'exact_resume'}
 write(out/'provenance.json',{'start_time':time.strftime('%Y-%m-%dT%H:%M:%S'),'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'sources').glob('*.py')},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'train_manifest':train.manifest,'validation_manifest':val.manifest,'anchor_manifest':getattr(train,'anchor_manifest',None),'parameter_count':sum(p.numel() for p in agent.parameters() if p.requires_grad),'parent':parent,'guidance_provenance':guidance_provenance,'clinical_ready':False})
 start=time.time();initial_step=step;log=(out/'history.jsonl').open('a',buffering=1);window=[]
 try:
  while step<cfg['updates']:
   if cursor==train.size:full_passes+=1;permutation=torch.randperm(train.size,device='cuda');cursor=0
   ids=permutation[cursor:min(cursor+cfg['batch_size'],train.size)];cursor+=len(ids);b=train.batch(ids)
   step+=1;seen+=len(ids);td_seen+=int(b['q_valid'].sum());info=agent.update(b,step)
   if not all(np.isfinite(v) for v in info.values()):raise RuntimeError('Nonfinite training diagnostic')
   window.append(info)
   if step%cfg['log_every']==0:
    keys=set().union(*(x.keys() for x in window));item={k:float(np.mean([x[k] for x in window if k in x])) for k in keys}
    item.update(event='train',step=step,seen=seen,td_seen=td_seen,full_passes=full_passes+int(cursor==train.size),equivalent_dataset_passes=seen/train.size,elapsed_seconds=time.time()-start,updates_per_second=(step-initial_step)/(time.time()-start),gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
    log.write(json.dumps(item,allow_nan=False)+'\n');print(json.dumps(item,allow_nan=False),flush=True);window=[]
   if step%cfg['save_every']==0 or step==cfg['updates']:
    checkpoint={'agent':agent.state_dict(),'optimizers':{k:v.state_dict() for k,v in optimizers.items()},'config':cfg,'step':step,'seen':seen,'td_seen':td_seen,'full_passes':full_passes,'cursor':cursor,'permutation':permutation,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
    torch.save(checkpoint,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
    torch.save({'agent':agent.state_dict(),'config':cfg,'step':step,'clinical_ready':False},out/('policy_%07d.pt'%step))
    ev=diagnostic(agent,val);write(out/('diagnostic_%07d.json'%step),ev);print(json.dumps({'event':'diagnostic','step':step,**ev}),flush=True)
  write(out/'completion.json',{'status':'completed_budgeted_policy_training','step':step,'new_updates':step-initial_step,'seen':seen,'td_seen':td_seen,'full_passes':full_passes+int(cursor==train.size),'equivalent_dataset_passes':seen/train.size,'wall_seconds':time.time()-start,'clinical_ready':False,'policy_evaluation_pending':True})
 except Exception as e:
  write(out/'failure.json',{'step':step,'error':str(e),'traceback':traceback.format_exc(),'clinical_ready':False});raise

if __name__=='__main__':main()
