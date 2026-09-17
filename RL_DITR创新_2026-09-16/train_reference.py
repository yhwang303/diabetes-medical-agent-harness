"""Two-stage, single-seed RL-DITR task adaptation with complete experiment records."""
import argparse,copy,hashlib,json,random,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from ditr_model import DITRAgent,status_score
from ditr_data import Data,tensor,ROOT
from ditr_losses import patient_loss,policy_loss,masked_mean

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

@torch.no_grad()
def evaluate(agent,data,per_patient=32):
 agent.eval();rng=np.random.default_rng(260915);stats={};patients=[]
 for pid,d in data.patients.items():
  positions=np.sort(rng.choice(len(d['starts']),min(per_patient,len(d['starts'])),replace=False));b=tensor(data.batch(pid,positions,12))
  # Torch2.0 fused FP32 attention produces NaN on a verified finite masked case.
  with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):out=agent.patient.rollout(b['state'],b['action'])
  local={}
  for h in [1,3,6,12]:
   valid=b['mask'][:,h-1];err=out['glucose'][valid,h-1]-b['target'][valid,h-1]
   last=d['glucose'][b['row'].cpu().numpy()];last=torch.tensor(last,device='cuda');le=last[valid]-b['target'][valid,h-1]
   if len(err):local[str(h*5)]={'n':len(err),'sse':float(err.square().sum()),'sae':float(err.abs().sum()),'last_sse':float(le.square().sum())}
  rs=masked_mean((out['reward']*12-status_score(b['target'])).square(),b['mask'])
  mu,std=agent.policy.parameters_at(out['memory'][:,-1]);policy_mse=(mu-b['action'][:,0]).square().mean()
  patients.append({'patient':pid,'horizons':local,'reward_status_mse':float(rs),'policy_action_mse':float(policy_mse),'mean_policy_std':float(std.mean()),'mean_value':float(out['initial_value'].mean())})
 for h in ['5','15','30','60']:
  values=[p['horizons'][h] for p in patients if h in p['horizons']];n=sum(v['n'] for v in values)
  stats[h]={'n':n,'rmse_mmol_l':(sum(v['sse'] for v in values)/n)**.5,'mae_mmol_l':sum(v['sae'] for v in values)/n,'last_rmse_mmol_l':(sum(v['last_sse'] for v in values)/n)**.5}
 result={'patients':patients,'horizons_minutes':stats,'reward_status_mse':float(np.mean([p['reward_status_mse'] for p in patients])),'policy_action_mse':float(np.mean([p['policy_action_mse'] for p in patients])),'per_patient':per_patient,'selection':'patient checkpoint: RMSE30 + RMSE60; policy checkpoints retained for independent development control evaluation, not selected by action imitation'}
 # Fail explicitly instead of silently picking a nonfinite checkpoint.
 json.dumps(result,allow_nan=False)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());assert cfg['seed']==260915
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=True
 out=ROOT/'results'/cfg['name'];out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
 sources=out/'sources';sources.mkdir()
 for p in ROOT.glob('*.py'):shutil.copy2(p,sources/p.name)
 train=Data('train');validation=Data('validation');assert train.total==1653421 and len(train.patients)==225
 write(out/'provenance.json',{'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'train_manifest':train.manifest,'validation_manifest':validation.manifest,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S')})
 agent=DITRAgent(**cfg['model']).cuda();target=copy.deepcopy(agent.patient).eval();target.requires_grad_(False)
 if cfg.get('initialize_checkpoint'):
  checkpoint=torch.load(ROOT/cfg['initialize_checkpoint']);assert checkpoint['stage'] in (['patient','policy'] if cfg.get('allow_policy_initialization') else ['patient']);agent.load_state_dict(checkpoint['agent']);target.load_state_dict(checkpoint['target_patient'])
  write(out/'initialization.json',{'checkpoint':cfg['initialize_checkpoint'],'sha256':hashlib.sha256((ROOT/cfg['initialize_checkpoint']).read_bytes()).hexdigest(),'source_stage':checkpoint['stage'],'source_step':checkpoint['step'],'optimizer_restarted':True,'training_seed':cfg['seed']})
 simulation=None;simulation_updates=0;simulation_arms=0
 if cfg.get('simulation'):
  from paired_data import PairedData,paired_path_loss
  simulation=PairedData();simulation_batches=simulation.batches(cfg['simulation']['groups_per_update'],cfg['seed']);write(out/'simulation_manifest.json',simulation.manifest)
 if cfg.get('rng_checkpoint'):
  rng_checkpoint=torch.load(ROOT/cfg['rng_checkpoint'],map_location='cpu');torch.set_rng_state(rng_checkpoint['rng_torch']);torch.cuda.set_rng_state_all(rng_checkpoint['rng_cuda']);np.random.set_state(rng_checkpoint['rng_numpy'])
  write(out/'rng_restoration.json',{'checkpoint':cfg['rng_checkpoint'],'purpose':'same-seed replay of interrupted original policy stage, not an independent repetition'})
 resume=torch.load(ROOT/cfg['resume_checkpoint']) if cfg.get('resume_checkpoint') else None
 reference_policy=None
 if cfg.get('policy_loss_variant')=='delayed_trajectory_kl':
  from delayed_policy import delayed_trajectory_loss
  assert cfg.get('stages')==['policy'] and cfg.get('initialize_checkpoint') and resume is None
  reference_policy=copy.deepcopy(agent.policy).eval();reference_policy.requires_grad_(False)
  write(out/'policy_reference.json',{'checkpoint':cfg['initialize_checkpoint'],'sha256':hashlib.sha256((ROOT/cfg['initialize_checkpoint']).read_bytes()).hexdigest(),'component':'agent.policy','frozen':True,'constraint':'soft sampled trajectory KL penalty, not a hard trust region','objective':cfg['delayed_policy']})
 log=(out/'history.jsonl').open('a',buffering=1);start=time.time();stage='initialization';step=0
 def event(obj):
  obj['elapsed_seconds']=time.time()-start;line=json.dumps(obj,allow_nan=False);log.write(line+'\n');print(line,flush=True)
 def save(path,stage,step,samples,optimizer):
  torch.save({'agent':agent.state_dict(),'target_patient':target.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'stage':stage,'step':step,'samples':samples,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state()},path)
 try:
  stages=cfg.get('stages',['patient','policy'])
  for stage in stages:
   if stage=='policy' and cfg.get('policy_rng_seed') is not None:
    torch.manual_seed(cfg['policy_rng_seed']);event({'event':'policy_rng_reset','seed':cfg['policy_rng_seed'],'purpose':'same policy sampling stream for matched initialization ablation'})
   stagecfg=cfg[stage];agent.patient.requires_grad_(stage=='patient');agent.policy.requires_grad_(stage=='policy')
   optimizer=torch.optim.Adam(agent.patient.parameters() if stage=='patient' else agent.policy.parameters(),lr=stagecfg['lr'],weight_decay=cfg['weight_decay'])
   step=0;samples=0;visited=set();best=float('inf');window=[];last=time.time();full_epochs=0;skip_steps=0
   if stage=='patient' and resume is not None:
    assert resume['stage']=='patient'
    # Re-evaluate saved pre-failure candidates under the corrected numerical backend.
    for candidate in sorted((ROOT/cfg['resume_checkpoint']).parent.glob('patient_0*.pt')):
     previous=torch.load(candidate);agent.load_state_dict(previous['agent']);ev=evaluate(agent,validation,cfg['validation_per_patient']);score=ev['horizons_minutes']['30']['rmse_mmol_l']+ev['horizons_minutes']['60']['rmse_mmol_l']
     write(out/('revalidated_%06d.json'%previous['step']),ev)
     if score<best:best=score;torch.save(previous,out/'patient_best.pt');write(out/'patient_best_evaluation.json',ev)
    agent.load_state_dict(resume['agent']);target.load_state_dict(resume['target_patient']);optimizer.load_state_dict(resume['optimizer']);step=resume['step'];samples=resume['samples'];skip_steps=step
    torch.set_rng_state(resume['rng_torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in resume['rng_cuda']]);np.random.set_state(resume['rng_numpy'])
    event({'event':'resume_same_seed_and_optimizer','stage':stage,'step':step,'samples':samples,'checkpoint':cfg['resume_checkpoint']})
   event({'event':'stage_start','stage':stage,'patients':len(train.patients),'origins':train.total})
   for epoch in range(stagecfg['epochs']):
    completed=True
    for batch_index,raw in enumerate(train.batches(cfg['batch_size'],cfg['seed']+epoch,cfg['horizon'])):
     if epoch==0 and batch_index<skip_steps:visited.add(raw['patient']);continue
     b=tensor(raw);agent.train();agent.patient.train(stage=='patient');optimizer.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['amp']):
      if stage=='policy' and reference_policy is not None:
       loss,parts=delayed_trajectory_loss(agent,reference_policy,b['state'],**cfg['delayed_policy'])
      else:
       loss,parts=patient_loss(agent.patient,target,b,cfg['mu']) if stage=='patient' else policy_loss(agent,b,cfg['horizon'],variant=cfg.get('policy_loss_variant','legacy'))
      if stage=='patient' and simulation is not None:
       parts.update({k:loss.new_zeros(()) for k in ['simulation_update','simulation_factual_loss','paired_glucose_effect_mse','paired_status_effect_mse']})
       if (step+1)%cfg['simulation']['every_steps']==0:
        sb=tensor(next(simulation_batches));agent.patient.eval()
        # Identical shared-state arms must not differ because of dropout masks.
        factual,_,prediction=patient_loss(agent.patient,target,sb,cfg['mu'],return_rollout=True)
        paired,pair_parts=paired_path_loss(prediction,sb)
        loss=loss+cfg['simulation']['factual_weight']*factual+cfg['simulation']['paired_weight']*paired
        parts.update({'simulation_update':loss.new_ones(()),'simulation_factual_loss':factual,**pair_parts});simulation_updates+=1;simulation_arms+=len(sb['state']);agent.patient.train()
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite '+stage+' loss')
     loss.backward();grad=torch.nn.utils.clip_grad_norm_(agent.patient.parameters() if stage=='patient' else agent.policy.parameters(),5.)
     if not torch.isfinite(grad):raise RuntimeError('Nonfinite '+stage+' gradient')
     optimizer.step()
     if stage=='patient':
      with torch.no_grad():
       for dst,src in zip(target.parameters(),agent.patient.parameters()):dst.lerp_(src,cfg['target_tau'])
     step+=1;samples+=len(raw['state']);visited.add(raw['patient']);window.append({'loss':float(loss),**{k:float(v) for k,v in parts.items()}})
     if step%cfg['log_every']==0:
      now=time.time();event({'event':'train','stage':stage,'step':step,'samples':samples,'patients_seen':len(visited),'seconds_per_step':(now-last)/len(window),'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,**{k:float(np.mean([w[k] for w in window])) for k in window[0]}});window=[];last=now
     if step%stagecfg['eval_every']==0:
      save(out/(stage+'_%06d.pt'%step),stage,step,samples,optimizer);ev=evaluate(agent,validation,cfg['validation_per_patient']);write(out/(stage+'_eval_%06d.json'%step),ev)
      score=ev['horizons_minutes']['30']['rmse_mmol_l']+ev['horizons_minutes']['60']['rmse_mmol_l'];event({'event':'validation','stage':stage,'step':step,'rmse30_plus60':score,'reward_status_mse':ev['reward_status_mse'],'policy_action_mse':ev['policy_action_mse']})
      if stage=='patient' and score<best:best=score;save(out/'patient_best.pt',stage,step,samples,optimizer);write(out/'patient_best_evaluation.json',ev)
      last=time.time()
     if step>=stagecfg['max_steps']:completed=False;break
    if completed:full_epochs+=1
    if not completed:break
   save(out/(stage+'_last.pt'),stage,step,samples,optimizer);ev=evaluate(agent,validation,cfg['final_validation_per_patient']);write(out/(stage+'_final_evaluation.json'),ev)
   score=ev['horizons_minutes']['30']['rmse_mmol_l']+ev['horizons_minutes']['60']['rmse_mmol_l']
   # Use identical fixed validation sample for selection; larger final evaluation is diagnostic only.
   selected_ev=evaluate(agent,validation,cfg['validation_per_patient']);selected_score=selected_ev['horizons_minutes']['30']['rmse_mmol_l']+selected_ev['horizons_minutes']['60']['rmse_mmol_l']
   if stage=='patient' and selected_score<best:best=selected_score;save(out/'patient_best.pt',stage,step,samples,optimizer);write(out/'patient_best_evaluation.json',selected_ev)
   write(out/(stage+'_completion.json'),{'steps':step,'samples_seen':samples,'patients_seen':len(visited),'full_epochs':full_epochs,'all_origins':train.total,'selection_score':best if stage=='patient' else None,'simulation_updates':simulation_updates if stage=='patient' else 0,'simulation_arm_visits':simulation_arms if stage=='patient' else 0})
   event({'event':'stage_complete','stage':stage,'steps':step,'samples':samples,'full_epochs':full_epochs})
   if stage=='patient':
    checkpoint=torch.load(out/'patient_best.pt');agent.load_state_dict(checkpoint['agent']);target.load_state_dict(checkpoint['target_patient']);event({'event':'patient_selected_for_policy','step':checkpoint['step'],'samples':checkpoint['samples']})
  status='two_stage_training_completed_control_evaluation_pending' if stages==['patient','policy'] else '_and_'.join(stages)+'_budget_completed_control_evaluation_pending'
  write(out/'completion.json',{'status':status,'stages':stages,'elapsed_seconds':time.time()-start,'clinical_ready':False});event({'event':'training_complete'})
 except Exception as error:
  write(out/'failure.json',{'stage':stage,'step':step,'error':repr(error),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise

if __name__=='__main__':main()
