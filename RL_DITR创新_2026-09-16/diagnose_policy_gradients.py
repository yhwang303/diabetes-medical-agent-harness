"""Separate imagined RL and behavior-fitting gradients on one fixed train sample."""
import hashlib,json
import numpy as np
import torch
from ditr_data import ROOT,Data,tensor
from ditr_model import DITRAgent
from ditr_losses import policy_loss

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 data=Data('train');rng=np.random.default_rng(260915);raw=[]
 for pid,d in data.patients.items():raw.append(data.batch(pid,np.sort(rng.choice(len(d['starts']),min(4,len(d['starts'])),replace=False)),12))
 keys=['state','action','target','mask','final_state'];batch=tensor({k:np.concatenate([r[k] for r in raw]) for k in keys});rows=[]
 for run in ['H02_prefix_sim_factual','H04_response_patient_rl','R07_H02_extra_policy_control']:
  path=ROOT/'results'/run/'policy_last.pt'
  if not path.exists():continue
  ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.patient.requires_grad_(False);torch.manual_seed(260915)
  _,parts=policy_loss(agent,batch,12,variant='bounded_return_nll');parameters=list(agent.policy.parameters());grads={}
  for key in ['historical_policy_loss','imagined_policy_loss','supervised_training_loss']:
   grad=torch.autograd.grad(parts[key],parameters,retain_graph=True);grads[key]=torch.cat([g.flatten() for g in grad]);assert torch.isfinite(grads[key]).all()
  norms={k:float(v.norm()) for k,v in grads.items()};behavior=grads['historical_policy_loss']+grads['supervised_training_loss'];imagined=grads['imagined_policy_loss'];den=behavior.norm()*imagined.norm()
  rows.append({'run':run,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'losses':{k:float(parts[k]) for k in grads},'gradient_l2':norms,'imagined_over_behavior_gradient_norm':float(imagined.norm()/behavior.norm().clamp_min(1e-12)),'imagined_behavior_cosine':float((imagined*behavior).sum()/den.clamp_min(1e-12)),'imagination_minutes':60})
 result={'scope':'training diagnostic, no optimizer updates; fixed4 origins per225 patients and common action-sampling seed; does not by itself prove which objective improves control','patients':len(raw),'origins':len(batch['state']),'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'results':rows};(ROOT/'checks/policy_gradient_balance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
