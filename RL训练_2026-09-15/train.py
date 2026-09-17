"""Tracked patient-model experiments; validation only for selection, sealed test never opened."""
from pathlib import Path
import argparse,json,time,random,hashlib,os,shutil
import numpy as np
import torch
import torch.nn.functional as F
from model import PatientModel
from data import Data,tensor,ROOT

def write_json(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def losses(model,b,mu=.1):
 glu,wtr,z,_=model.rollout(b['state'],b['action'])
 mask=b['mask']; target=b['target']; denom=mask.sum().clamp_min(1)
 gl=((glu-target).square()*mask).sum()/denom
 labels=((target>=3.9)&(target<=10)).long()
 cl=(F.cross_entropy(wtr.transpose(1,2),labels,reduction='none')*mask).sum()/denom
 target_z=model.encode(b['next_state'])[:,-1]
 consistency=(z[torch.arange(len(z),device=z.device),b['k']]-target_z).square().mean()
 return gl+cl+mu*consistency,{'glucose_mse':gl,'wtr_ce':cl,'consistency':consistency}

def metrics(y,p):
 e=np.array(p)-np.array(y)
 if not len(e) or not np.isfinite(e).all():raise ValueError('Invalid evaluation: empty or nonfinite predictions/targets')
 return {'n':int(len(e)),'mae_mmol_l':float(np.abs(e).mean()),'rmse_mmol_l':float(np.sqrt((e*e).mean()))}

@torch.no_grad()
def evaluate(model,data,per_patient=128,seed=260915,batch_size=128):
 model.eval();rng=np.random.default_rng(seed);records={h:{'y':[],'model':[],'last':[],'linear':[],'wtr':[],'class':[]} for h in [1,3,6,12]};patient=[];effects=[];strata={};preview=[];held={h:{'y':[],'model':[],'last':[],'linear':[]} for h in [6,12]}
 for pid,d in data.patients.items():
  # Fixed stratified-by-patient diagnostic sample, uniform over eligible starts; no fitting here.
  positions=np.sort(rng.choice(len(d['starts']),min(per_patient,len(d['starts'])),replace=False)) if per_patient else rng.permutation(len(d['starts']))
  local={h:{'y':[],'model':[],'last':[],'linear':[]} for h in records}
  for offset in range(0,len(positions),batch_size):
   bb=data.batch(pid,positions[offset:offset+batch_size],12);b=tensor(bb)
   pred,wtr,_,_=model.rollout(b['state'],b['action']); pred=pred.float().cpu().numpy();
   if not np.isfinite(pred).all():
    torch.save({'model':model.state_dict(),'batch':b},ROOT/'results/nonfinite_evaluation_debug.pt');raise RuntimeError('Nonfinite evaluation for '+pid)
   cl=wtr.argmax(-1).cpu().numpy();s=bb['row'];cur=d['glucose'][s]
   history=data.history(d,s);raw=history[:,:,0]*2.9487731123159437+7.602434716830251;observed=history[:,:,5]>.5
   # Least squares trend over up to the last 30 min observed values; missing positions excluded.
   time_axis=np.arange(-6,1,dtype=np.float32);m=observed[:,-7:];den=m.sum(1).clip(1);tm=(m*time_axis).sum(1)/den;gm=(m*raw[:,-7:]).sum(1)/den
   num=(m*(time_axis-tm[:,None])*(raw[:,-7:]-gm[:,None])).sum(1);var=(m*(time_axis-tm[:,None])**2).sum(1);slope=np.divide(num,var,out=np.zeros_like(num),where=var>0)
   for h in records:
    valid=bb['mask'][:,h-1];y=bb['target'][valid,h-1];pm=pred[valid,h-1];last=cur[valid];lin=(cur+slope*h)[valid]
    for key,value in [('y',y),('model',pm),('last',last),('linear',lin)]:records[h][key].extend(value.tolist());local[h][key].extend(value.tolist())
    if h in held:
     hold=valid & np.all(np.isclose(bb['action'][:,:h],bb['action'][:,:1],atol=1e-7,rtol=1e-7),axis=1)
     for key,value in [('y',bb['target'][hold,h-1]),('model',pred[hold,h-1]),('last',cur[hold]),('linear',(cur+slope*h)[hold])]:held[h][key].extend(value.tolist())
    records[h]['wtr'].extend(cl[valid,h-1].tolist());records[h]['class'].extend(((y>=3.9)&(y<=10)).tolist())
    if h==12:
     for name,condition in [('initial_low',cur<3.9),('initial_high',cur>10),('no_food_record_history',~(history[:,:,8]>.5).any(1)),('no_exercise_record_history',~(history[:,:,9]>.5).any(1)),('future_bolus_record',np.isfinite(d['bolus'][np.minimum(s[:,None]+np.arange(12),len(d['bolus'])-1)]).any(1))]:
      select=valid&condition
      if select.any():
       item=strata.setdefault(name,{'y':[],'model':[],'last':[]});item['y'].extend(bb['target'][select,11].tolist());item['model'].extend(pred[select,11].tolist());item['last'].extend(cur[select].tolist())
   # Paired counterfactual response diagnostic: same history, held basal +/-0.5 U/h.
   if offset==0:
    n=min(16,len(cur));st=b['state'][:n];a=b['action'][:n,:1];low=(a-.5).clamp(0,19);high=low+1
    lo=model.rollout(st,low.expand(-1,12))[0];hi=model.rollout(st,high.expand(-1,12))[0]
    delta=(hi-lo).float().cpu().numpy()
    for j in range(n):effects.append({'patient':pid,'row':int(s[j]),'low_u_h':float(low[j]),'high_u_h':float(high[j]),'delta_5':float(delta[j,0]),'delta_30':float(delta[j,5]),'delta_60':float(delta[j,11]),'initial_glucose':float(cur[j])})
    if len(preview)<6:preview.append({'patient':pid,'row':int(s[0]),'observed':bb['target'][0].tolist(),'valid':bb['mask'][0].tolist(),'predicted':pred[0].tolist()})
  patient.append({'patient':pid,'horizons':{str(h*5):{k:metrics(v['y'],v[k]) for k in ['model','last','linear']} for h,v in local.items() if v['y']}})
 summary={str(h*5):{**{k:metrics(v['y'],v[k]) for k in ['model','last','linear']},'wtr_accuracy':float(np.mean(np.array(v['wtr'])==np.array(v['class'])))} for h,v in records.items() if v['y']}
 for h in summary:
  # Patient bootstrap on MAE differences prevents overlapping windows masquerading as independent patients.
  dif=np.array([p['horizons'][h]['model']['mae_mmol_l']-p['horizons'][h]['last']['mae_mmol_l'] for p in patient if h in p['horizons']]);rs=np.random.default_rng(921)
  means=np.mean(rs.choice(dif,(2000,len(dif)),replace=True),1)
  summary[h]['patient_mean_model_minus_last_mae']=float(dif.mean());summary[h]['patient_bootstrap95_model_minus_last_mae']=np.quantile(means,[.025,.975]).tolist()
 delta=np.array([x['delta_60'] for x in effects]);sens={'samples':len(effects),'fraction_negative_60':float((delta<0).mean()),'median_delta60_per_plus1u_h':float(np.median(delta)),'mean_delta60_per_plus1u_h':float(delta.mean()),'mean_delta5_per_plus1u_h':float(np.mean([x['delta_5'] for x in effects]))}
 gate_b=all(h in summary and summary[h]['model']['rmse_mmol_l']<min(summary[h][k]['rmse_mmol_l'] for k in ['last','linear']) for h in ['30','60'])
 gate_a=sens['fraction_negative_60']>=.8 and -.02>sens['median_delta60_per_plus1u_h']>-3
 held_summary={str(h*5):{k:metrics(v['y'],v[k]) for k in ['model','last','linear']} for h,v in held.items() if v['y']}
 result={'evaluation_scope':'retrospective logged-action-conditioned validation; not online forecast or independent policy evaluation','held_action_subset':held_summary,'split':data.split,'per_patient_limit':per_patient,'seed':seed,'horizons_minutes':summary,'patients':patient,'strata':{n:{k:metrics(v['y'],v[k]) for k in ['model','last']} for n,v in strata.items()},'action_sensitivity':sens,'paired_action_effects':effects,'preview':preview,'gate_prediction_beats_both_baselines_30_60':gate_b,'gate_direction_sanity_only':gate_a,'gates_are_research_screens_not_counterfactual_validation':True,'clinical_ready':False}
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--eval-only',action='store_true');args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
 out=ROOT/'results'/cfg['name'];out.mkdir(exist_ok=True,parents=True)
 if not args.eval_only:
  if (out/'history.jsonl').exists():raise RuntimeError('Existing experiment name: use a new name to retain all attempts')
  write_json(out/'config.json',cfg)
  (out/'sources').mkdir(exist_ok=True)
  for file in ROOT.glob('*.py'):shutil.copy2(file,out/'sources'/file.name)
 model=PatientModel(**cfg['model']).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')};write_json(out/('evaluation_provenance.json' if args.eval_only else 'provenance.json'),{'source_hashes':source,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'train_manifest':json.loads((ROOT/'packed/train/manifest.json').read_text()),'validation_manifest':json.loads((ROOT/'packed/validation/manifest.json').read_text()),'start_time':time.strftime('%Y-%m-%dT%H:%M:%S')})
 if cfg.get('resume'):
  ck=torch.load(ROOT/'results'/cfg['resume']/'best.pt',map_location='cuda');model.load_state_dict(ck['model'])
 validation=Data('validation');start=time.time();best=float('inf');step=0;samples=0;visited=set();full_epochs=0
 if args.eval_only:
  ck=torch.load(out/'best.pt',map_location='cuda');model.load_state_dict(ck['model']);ev=evaluate(model,validation,cfg.get('final_eval_per_patient',0));write_json(out/'full_validation_evaluation.json',ev);return
 train=Data('train');log=(out/'history.jsonl').open('a',buffering=1)
 print(json.dumps({'event':'start','name':cfg['name'],'samples':train.total,'patients':len(train.patients),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
 last=time.time();window=[];stop=False
 for epoch in range(cfg['epochs']):
  completed=True
  for raw in train.batches(cfg['batch_size'],cfg['seed']+epoch,cfg['unroll']):
   model.train();b=tensor(raw);optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=cfg.get('amp',True)):
    loss,parts=losses(model,b,cfg['mu'])
   if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
   if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
   optimizer.step();step+=1;samples+=len(raw['state']);visited.add(raw['patient']);window.append([float(loss),*[float(x) for x in parts.values()]])
   if step%cfg.get('log_every',25)==0:
    now=time.time();values=np.mean(window,0).tolist();item={'event':'train','step':step,'epoch':epoch,'samples_seen':samples,'patients_seen':len(visited),'loss':values[0],'glucose_mse':values[1],'wtr_ce':values[2],'consistency':values[3],'seconds_per_step':(now-last)/len(window),'elapsed_seconds':now-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9};print(json.dumps(item),flush=True);log.write(json.dumps(item)+'\n');last=now;window=[]
   if step%cfg['eval_every']==0 or step>=cfg['max_steps']:
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'samples_seen':samples},out/'pre_evaluation.pt')
    ev=evaluate(model,validation,cfg['eval_per_patient']);score=ev['horizons_minutes']['30']['model']['rmse_mmol_l']+ev['horizons_minutes']['60']['model']['rmse_mmol_l'];write_json(out/('eval_%06d.json'%step),ev)
    item={'event':'eval','step':step,'score_rmse30_plus60':score,'gate_prediction':ev['gate_prediction_beats_both_baselines_30_60'],'gate_direction':ev['gate_direction_sanity_only']};print(json.dumps(item),flush=True);log.write(json.dumps(item)+'\n')
    ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'samples_seen':samples,'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state()};torch.save(ck,out/'last.pt')
    if score<best:best=score;torch.save(ck,out/'best.pt');write_json(out/'best_evaluation.json',ev)
    last=time.time()
   if step>=cfg['max_steps'] or time.time()-start>cfg['max_seconds']:stop=True;completed=False;break
  if completed:full_epochs+=1
  if stop:break
 # Evaluate the final iterate too; do not silently omit the tail of a full epoch.
 ev=evaluate(model,validation,cfg['eval_per_patient']);score=ev['horizons_minutes']['30']['model']['rmse_mmol_l']+ev['horizons_minutes']['60']['model']['rmse_mmol_l']
 write_json(out/('eval_%06d.json'%step),ev)
 ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'samples_seen':samples,'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state()}
 torch.save(ck,out/'last.pt')
 if score<best:
  best=score;torch.save(ck,out/'best.pt');write_json(out/'best_evaluation.json',ev)
 if not (out/'best.pt').exists():
  torch.save({'model':model.state_dict(),'step':step,'config':cfg},out/'best.pt')
 ck=torch.load(out/'best.pt',map_location='cuda');model.load_state_dict(ck['model']);ev=evaluate(model,validation,cfg.get('final_eval_per_patient',256));write_json(out/'final_evaluation.json',ev)
 write_json(out/'completion.json',{'status':'completed_budgeted_experiment','steps':step,'samples_seen':samples,'all_eligible_train_samples':train.total,'patients_seen':len(visited),'full_epochs':full_epochs,'best_checkpoint_step':ck['step'],'wall_seconds':time.time()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'gate_prediction':ev['gate_prediction_beats_both_baselines_30_60'],'gate_direction':ev['gate_direction_sanity_only'],'clinical_ready':False});print('COMPLETE',cfg['name'],flush=True)
if __name__=='__main__':main()
