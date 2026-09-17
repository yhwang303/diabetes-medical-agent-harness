"""Bounded mechanism experiment on existing data before integrating into RL-DITR.
No policy/control result is claimed by this response-only experiment.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent
from paired_data import PairedData
from response_operator import ResponseOperator,DirectPrefixResponse

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def metrics(pred,true,mask):
 result={}
 for h in [6,12,24,48]:
  valid=mask[...,h-1];dp=pred[...,h-1][valid]*18;dt=true[...,h-1][valid]*18;meaningful=dt.abs()>1
  result[str(h*5)]={'pairs':int(valid.sum()),'effect_mae_mg_dl':float((dp-dt).abs().mean()),'true_effect_rms_mg_dl':float(dt.square().mean().sqrt()),'predicted_effect_rms_mg_dl':float(dp.square().mean().sqrt()),'direction_pairs_above_1mg_dl':int(meaningful.sum()),'direction_agreement':float((dp[meaningful].sign()==dt[meaningful].sign()).float().mean()) if meaningful.any() else None}
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--global-kernel',action='store_true');ap.add_argument('--direct-prefix',action='store_true');ap.add_argument('--training-data',choices=['paired_sim_train','paired_sim_train_temporal'],default='paired_sim_train');args=ap.parse_args()
 assert not (args.global_kernel and args.direct_prefix)
 cfg={'name':args.name,'seed':260915,'epochs':100,'groups_per_batch':16,'learning_rate':.001,'weight_decay':.0001,'basis_degree':7,'response_hidden':64,'conditioned':not args.global_kernel,'encoder_checkpoint':'results/H02_prefix_sim_factual/patient_best.pt','training_data':'paired_sim_train','scope':'response-operator mechanism prototype only; all Loop data already used by frozen encoder, no Loop retraining in this stage; not an RL controller or established innovation'}
 if args.direct_prefix:cfg.update({'architecture':'direct_prefix_response','response_hidden':54,'capacity_control':'17469 versus P01 17480 parameters; same data, epochs, batches, optimizer and seed; causal prefix but no convolution restriction'})
 cfg['training_data']=args.training_data
 if args.training_data=='paired_sim_train_temporal':cfg['scope']='response-only temporal-data ablation; 504 same-state groups, seven time-varying arms, 100epochs/3200updates; no policy or control claims; not a pure data-size-matched comparison with constant five-arm runs'
 out=ROOT/'results'/args.name;out.mkdir(exist_ok=False);write(out/'config.json',cfg)
 for name in ['train_response_operator.py','response_operator.py']:shutil.copy2(ROOT/name,out/name)
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 checkpoint=ROOT/cfg['encoder_checkpoint'];ck=torch.load(checkpoint,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 data=PairedData(args.training_data);x=data.arrays['features'][:,:,0:72];clock=np.broadcast_to(data.clock,(len(x),72,2));history=np.concatenate([x[:,0],clock],-1);context=[]
 with torch.no_grad():
  for offset in range(0,len(history),32):context.append(agent.patient.encode(torch.tensor(history[offset:offset+32],device='cuda'))[:,-1])
 context=torch.cat(context);actions=torch.tensor(data.arrays['action'],device='cuda');target=torch.tensor(data.arrays['target'],device='cuda');mask=torch.tensor(data.arrays['mask'],device='cuda');da=actions[:,1:]-actions[:,0:1];dy=target[:,1:]-target[:,0:1];joint=mask[:,1:]&mask[:,0:1]
 model=(DirectPrefixResponse(hidden=cfg['response_hidden'],degree=cfg['basis_degree']) if args.direct_prefix else ResponseOperator(hidden=cfg['response_hidden'],degree=cfg['basis_degree'],conditioned=cfg['conditioned'])).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);log=(out/'history.jsonl').open('w',buffering=1);start=time.time();steps=0
 for epoch in range(cfg['epochs']):
  order=np.random.default_rng(cfg['seed']+epoch).permutation(len(context));losses=[]
  for offset in range(0,len(order),cfg['groups_per_batch']):
   idx=order[offset:offset+cfg['groups_per_batch']];pred=model(context[idx],da[idx]);valid=joint[idx];loss=((pred-dy[idx]).square()*valid).sum()/valid.sum();assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);optimizer.step();steps+=1;losses.append(float(loss))
  if (epoch+1)%10==0:
   event={'epoch':epoch+1,'steps':steps,'training_effect_mse_mmol2':float(np.mean(losses)),'elapsed_seconds':time.time()-start};log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
 model.eval()
 with torch.no_grad():
  training=metrics(model(context,da),dy,joint);diagnostics={}
  for folder in ['action_probe','action_probe_dynamic']:
   cases=json.loads((ROOT/folder/'samples.json').read_text());predictions=[];truth=[];masks=[];timing=[]
   for case in cases:
    arms=case['arms'];history=torch.tensor(np.array(case['history'],dtype='float32')[None],device='cuda');z=agent.patient.encode(history)[:,-1];a=torch.zeros((1,len(arms),48),device='cuda');y=torch.zeros_like(a);valid=torch.zeros_like(a,dtype=torch.bool)
    for i,arm in enumerate(arms):
     n=len(arm['records']);a[0,i,:n]=torch.tensor([r['delivered_basal_u_h'] for r in arm['records']],device='cuda');y[0,i,:n]=torch.tensor([r['cgm']/18 for r in arm['records']],device='cuda');valid[0,i,:n]=True
    p=model(z,a-a[:,0:1]);predictions.append(p[:,1:]);truth.append(y[:,1:]-y[:,0:1]);masks.append(valid[:,1:]&valid[:,0:1])
    if folder.endswith('dynamic'):
     ids={arm['pattern']:i for i,arm in enumerate(arms)}
     for f,s in [('early_plus','late_plus'),('early_minus','late_minus'),('plus_then_minus','minus_then_plus')]:
      i,j=ids[f],ids[s]
      if valid[0,i,-1] and valid[0,j,-1]:timing.append({'patient':case['patient'],'minute':case['minute'],'plans':[f,s],'true_final_difference_mg_dl':float((y[0,i,-1]-y[0,j,-1])*18),'predicted_final_difference_mg_dl':float((p[0,i,-1]-p[0,j,-1])*18)})
   diagnostics[folder]=metrics(torch.cat(predictions),torch.cat(truth),torch.cat(masks))
   if timing:
    meaningful=[x for x in timing if abs(x['true_final_difference_mg_dl'])>1];diagnostics['equal_dose_timing']={'pairs':len(timing),'meaningful_pairs':len(meaningful),'direction_agreement':float(np.mean([np.sign(x['true_final_difference_mg_dl'])==np.sign(x['predicted_final_difference_mg_dl']) for x in meaningful])),'final_effect_mae_mg_dl':float(np.mean([abs(x['true_final_difference_mg_dl']-x['predicted_final_difference_mg_dl']) for x in timing])),'cases':timing}
 torch.save({'model':model.state_dict(),'config':cfg,'encoder_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},out/'response_last.pt')
 result={'status':'mechanism_experiment_completed_not_policy_or_control','training_seed':cfg['seed'],'steps':steps,'training_groups':data.total,'response_parameters':sum(p.numel() for p in model.parameters()),'encoder_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'training':training,'development':diagnostics,'no_development_checkpoint_selection':True,'elapsed_seconds':time.time()-start};write(out/'response_evaluation.json',result);write(out/'completion.json',{'status':result['status'],'elapsed_seconds':result['elapsed_seconds']});print(json.dumps({'training':training,'development':{k:v for k,v in diagnostics.items() if k!='equal_dose_timing'},'timing_summary':{k:v for k,v in diagnostics.get('equal_dose_timing',{}).items() if k!='cases'}}),flush=True)

if __name__=='__main__':main()
