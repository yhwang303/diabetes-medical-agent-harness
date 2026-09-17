"""Identify latent action response with a frozen full-data RL-DITR representation."""
import hashlib,json,shutil,time
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent
from paired_data import PairedData
from response_operator import StateResponseOperator

def main():
 cfg={'name':'P03_latent_response_operator','seed':260915,'epochs':100,'groups_per_batch':16,'learning_rate':.001,'weight_decay':.0001,'basis_degree':7,'latent_rank':16,'encoder_checkpoint':'results/H02_prefix_sim_factual/patient_best.pt','training_data':'paired_sim_train','target_steps':[1,6,12,24,48],'scope':'latent response mechanism; supervised differences of frozen observed encodings; not independently evaluated control'}
 out=ROOT/'results'/cfg['name'];out.mkdir(exist_ok=False);(out/'config.json').write_text(json.dumps(cfg,indent=2));shutil.copy2(__file__,out/'train_state_response.py');shutil.copy2(ROOT/'response_operator.py',out/'response_operator.py')
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=ROOT/cfg['encoder_checkpoint'];ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False);data=PairedData();x=data.arrays['features'];groups,arms=x.shape[:2];targets=[];context=[];start=time.time()
 with torch.no_grad():
  for h in [0]+cfg['target_steps']:
   history=x[:,:,h:h+72].reshape(-1,72,20);clock=np.broadcast_to(data.clock,(len(history),72,2));history=np.concatenate([history,clock],-1);encoded=[]
   for offset in range(0,len(history),32):encoded.append(agent.patient.encode(torch.tensor(history[offset:offset+32],device='cuda'))[:,-1])
   encoded=torch.cat(encoded).reshape(groups,arms,256)
   if h==0:context=encoded[:,0]
   else:targets.append(encoded[:,1:]-encoded[:,0:1])
 y=torch.stack(targets,2);a=torch.tensor(data.arrays['action'],device='cuda');da=a[:,1:]-a[:,0:1];mask=torch.tensor(data.arrays['mask'],device='cuda');index=torch.tensor(cfg['target_steps'],device='cuda')-1;valid=(mask[:,1:]&mask[:,0:1])[:,:,index]
 model=StateResponseOperator(rank=cfg['latent_rank'],degree=cfg['basis_degree']).cuda();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);log=(out/'history.jsonl').open('w',buffering=1);steps=0
 for epoch in range(cfg['epochs']):
  order=np.random.default_rng(cfg['seed']+epoch).permutation(groups);losses=[]
  for offset in range(0,groups,cfg['groups_per_batch']):
   idx=order[offset:offset+cfg['groups_per_batch']];pred=model(context[idx],da[idx])[:,:,index];error=(pred-y[idx]).square().mean(-1);loss=(error*valid[idx]).sum()/valid[idx].sum();assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);optimizer.step();steps+=1;losses.append(float(loss))
  if (epoch+1)%10==0:
   event={'epoch':epoch+1,'steps':steps,'latent_effect_mse':float(np.mean(losses)),'elapsed_seconds':time.time()-start};log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
 with torch.no_grad():
  parts=[]
  for offset in range(0,groups,32):parts.append(model(context[offset:offset+32],da[offset:offset+32])[:,:,index])
  pred=torch.cat(parts);error=(pred-y).square().mean(-1);zero=y.square().mean(-1);mse=float((error*valid).sum()/valid.sum());null=float((zero*valid).sum()/valid.sum())
 torch.save({'model':model.state_dict(),'config':cfg,'encoder_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},out/'state_response_last.pt');result={'status':'latent_mechanism_completed_not_control','steps':steps,'training_groups':groups,'training_latent_effect_mse':mse,'zero_response_mse':null,'fraction_mse_reduction':1-mse/null,'elapsed_seconds':time.time()-start,'no_development_checkpoint_selection':True};(out/'state_response_evaluation.json').write_text(json.dumps(result,indent=2));(out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
