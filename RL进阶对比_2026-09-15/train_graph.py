"""Full-origin supervised dynamics + prior ranking; efficacy requires independent policy tests."""
from pathlib import Path
import json,argparse,time,hashlib,shutil,traceback
import torch
import numpy as np
from dynamics_data import Dynamics,ROOT
from graph_model import GraphPatient,patient_losses

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))
@torch.no_grad()
def evaluate(model,data,ids):
 sums={h:[0.,0.,0] for h in [0,2,5,11]};effect=[];model.eval()
 for start in range(0,len(ids),256):
  b=data.batch(ids[start:start+256]);z=model.encode(b['state']);pred=model.rollout(z,b['actions'],b['events']);err=(pred-b['target'])*data.scale
  for h in sums:
   valid=b['mask'][:,h];e=err[valid,h];sums[h][0]+=float(e.abs().sum());sums[h][1]+=float(e.square().sum());sums[h][2]+=len(e)
  keep=b['mask'][:,-1]
  if keep.any():
   base=b['actions'][:,0:1].expand(-1,12);p0=model.rollout(z,base,b['events']);p1=model.rollout(z,(base+.1).clamp(max=20),b['events']);effect.extend(((p1-p0)[keep,-1]*data.scale).tolist())
 metrics={str((h+1)*5):{'mae_mmol_l':x[0]/x[2],'rmse_mmol_l':(x[1]/x[2])**.5,'n':x[2]} for h,x in sums.items() if x[2]};model.train()
 return {'horizons':metrics,'plus_point1_basal_delta60_median':float(np.median(effect)) if effect else None,'fraction_negative60':float(np.mean(np.array(effect)<0)) if effect else None,'scope':'retrospective logged common-event conditional prediction; ranking diagnostic is prior consistency only','clinical_ready':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=ROOT/'results'/cfg['name'];out.mkdir(parents=True,exist_ok=True)
 if (out/'provenance.json').exists():raise RuntimeError('Preserve existing run; choose another name')
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);train=Dynamics('train');val=Dynamics('validation');model=GraphPatient(cfg).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']);generator=torch.Generator(device='cuda').manual_seed(7711);valid_ids=torch.randperm(val.size,device='cuda',generator=generator)[:8192]
 write(out/'config.json',cfg);(out/'sources').mkdir(exist_ok=True)
 for file in ['graph_model.py','train_graph.py','dynamics_data.py','rl_data.py','prepare_dynamics.py']:shutil.copy2(ROOT/file,out/'sources'/file)
 write(out/'provenance.json',{'config':cfg,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'sources').glob('*.py')},'train_origins':train.size,'train_manifest':train.manifest,'dynamics_manifest':train.dynamics_manifest,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'clinical_ready':False})
 step=0;seen=0;best=float('inf');begin=time.time();log=(out/'history.jsonl').open('w',buffering=1);window=[];stop=False;completed_epochs=0
 try:
  for epoch in range(cfg['epochs']):
   order=torch.randperm(train.size,device='cuda')
   for cursor in range(0,train.size,cfg['batch_size']):
    ids=order[cursor:cursor+cfg['batch_size']];batch=train.batch(ids);optimizer.zero_grad(set_to_none=True);loss,info=patient_losses(model,batch,cfg['alpha'],cfg['phi']);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);assert torch.isfinite(grad);optimizer.step();step+=1;seen+=len(ids);window.append({**info,'loss':float(loss),'grad_norm':float(grad)})
    if step%cfg['log_every']==0:
     row={k:float(np.mean([x[k] for x in window])) for k in window[0]};row.update(step=step,seen=seen,epoch=epoch,elapsed_seconds=time.time()-begin);window=[];log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    final=cursor+cfg['batch_size']>=train.size;smoke_end=bool(cfg.get('max_steps') and step>=cfg['max_steps'])
    if step%cfg['eval_every']==0 or final or smoke_end:
     ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'step':step,'seen':seen,'epoch':epoch,'cursor_next':cursor+len(ids),'order':order,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()};torch.save(ck,out/'last.pt');ev=evaluate(model,val,valid_ids);write(out/('eval_%07d.json'%step),ev);score=ev['horizons']['30']['rmse_mmol_l']+ev['horizons']['60']['rmse_mmol_l'];print(json.dumps({'event':'validation','step':step,**ev}),flush=True)
     if score<best:best=score;torch.save({'model':model.state_dict(),'config':cfg,'step':step,'clinical_ready':False},out/'best.pt');write(out/'best_evaluation.json',ev)
    if smoke_end:stop=True;break
   if final:completed_epochs+=1
   if stop:break
  write(out/'completion.json',{'steps':step,'seen':seen,'full_epochs':completed_epochs,'equivalent_passes':seen/train.size,'wall_seconds':time.time()-begin,'best_prediction_score':best,'policy_evaluation_pending':True,'clinical_ready':False})
 except Exception as e:write(out/'failure.json',{'step':step,'error':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
