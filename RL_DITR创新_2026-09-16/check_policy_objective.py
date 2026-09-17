"""Verify the negative-return pathology and its bounded likelihood repair."""
import json
import numpy as np
import torch
from ditr_data import ROOT,Data,tensor
from ditr_model import DITRAgent,status_score
from ditr_losses import GAMMA,discounted_return,policy_loss

def main():
 torch.set_num_threads(4);torch.manual_seed(260915);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);bound=(1/12)/(1-GAMMA);analytic=[]
 for std in [.5,.1,.02]:
  logp=torch.distributions.Normal(torch.tensor(2.),torch.tensor(std)).log_prob(torch.tensor(1.));legacy=float(.5*logp+1);repaired=float(-(1-.5/bound)*logp);analytic.append({'std':std,'legacy_historical_plus_mse':legacy,'repaired_historical_plus_nll':repaired})
 assert analytic[-1]['legacy_historical_plus_mse']<analytic[0]['legacy_historical_plus_mse'];assert analytic[-1]['repaired_historical_plus_nll']>analytic[0]['repaired_historical_plus_nll']
 ck=torch.load(ROOT/'results/R04_recursive_sim_factual/patient_best.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda();agent.load_state_dict(ck['agent']);agent.patient.eval().requires_grad_(False);agent.policy.requires_grad_(True);data=Data('train');rng=np.random.default_rng(260915);returns=[];first=None
 with torch.no_grad():
  for pid,d in data.patients.items():
   positions=rng.choice(len(d['starts']),min(32,len(d['starts'])),replace=False);b=tensor(data.batch(pid,positions,12));end=agent.patient.value(agent.patient.encode(b['final_state'])[:,-1]).squeeze(-1);r=discounted_return(status_score(b['target'])/12,b['mask'],end);returns.extend(r.cpu().tolist())
   if first is None:first=b
 a=np.array(returns);weights=1+np.clip(a/bound,-1,1);assert weights.min()>=0 and weights.max()<=2
 with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=policy_loss(agent,first,variant='bounded_return_nll')
 loss.backward();grad=torch.nn.utils.clip_grad_norm_(agent.policy.parameters(),5.);assert torch.isfinite(loss) and torch.isfinite(grad);assert all(p.grad is None for p in agent.patient.parameters());assert any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.policy.parameters())
 result={'status':'passed','analytic_negative_return_example':analytic,'physical_return_bound':bound,'real_training_patients_sampled':len(data.patients),'real_training_origins_sampled':len(a),'negative_return_fraction':float(np.mean(a<0)),'return_min':float(a.min()),'effective_weight_range':[float(weights.min()),float(weights.max())],'policy_gradient_norm':float(grad),'patient_frozen':True,'loss_parts':{k:float(v) for k,v in parts.items()},'not_innovation_or_efficacy_evidence':True}
 (ROOT/'checks/bounded_policy_objective.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
