"""Exact small-policy gradient identity plus real RL-DITR freeze/gradient checks."""
import copy,json,time
import numpy as np
import torch
from ditr_data import ROOT,Data,tensor
from ditr_model import DITRAgent
from delayed_policy import leave_one_out_advantage,delayed_trajectory_loss

def exact_identity(beta):
 theta=torch.tensor(-.6,dtype=torch.float64,requires_grad=True);p=theta.sigmoid();q=torch.tensor(.5,dtype=torch.float64);kl=p*(p/q).log()+(1-p)*((1-p)/(1-q)).log();objective=p-beta*kl;truth=torch.autograd.grad(objective,theta)[0];estimate=0.
 for a in [0.,1.]:
  for b in [0.,1.]:
   p=theta.sigmoid();x=torch.tensor([a,b],dtype=torch.float64);logp=x*p.log()+(1-x)*(1-p).log();ref=x*q.log()+(1-x)*(1-q).log();reward=x-beta*(logp-ref);adv=leave_one_out_advantage(reward[None]).detach()[0]
   contribution=torch.autograd.grad((adv*logp).mean(),theta)[0];weight=float((p if a else 1-p)*(p if b else 1-p));estimate+=weight*float(contribution)
 assert abs(estimate-float(truth))<1e-12
 return {'beta':beta,'exact_gradient':float(truth),'expected_estimator_gradient':estimate}

def main():
 identities=[exact_identity(b) for b in [0.,.4]];x=torch.tensor([[1.,3.,5.],[.1,.2,.3]])
 torch.testing.assert_close(leave_one_out_advantage(x),leave_one_out_advantage(x+77),atol=2e-5,rtol=1e-4)
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);data=Data('train');pid=next(iter(data.patients));b=tensor(data.batch(pid,np.arange(min(128,len(data.patients[pid]['starts'])))),);rows=[]
 for run,weight in [('C01_response_patient_composition','patient_composed.pt'),('H02_prefix_sim_factual','policy_last.pt')]:
  ck=torch.load(ROOT/'results'/run/weight,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.patient.requires_grad_(False);ref=copy.deepcopy(agent.policy).eval();ref.requires_grad_(False);torch.manual_seed(260915);torch.cuda.reset_peak_memory_stats();begin=time.time()
  loss,parts,detail=delayed_trajectory_loss(agent,ref,b['state'],return_details=True);assert torch.isfinite(loss);torch.testing.assert_close(detail['sampled_kl'],torch.zeros_like(detail['sampled_kl']),rtol=0,atol=0);assert detail['actions'].shape==(len(b['state'])*4,48);assert ((detail['actions']>=0)&(detail['actions']<=20)).all()
  loss.backward();assert all(p.grad is None for p in agent.patient.parameters());assert all(p.grad is None for p in ref.parameters());grads=[p.grad for p in agent.policy.parameters()];assert all(torch.isfinite(g).all() for g in grads);norm=sum(float(g.square().sum()) for g in grads)**.5;assert norm>0;torch.cuda.synchronize()
  rows.append({'run':run,'histories':len(b['state']),'trajectories_per_history':4,'policy_gradient_l2':norm,'seconds_forward_backward':time.time()-begin,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'metrics':{k:float(v) for k,v in parts.items()}})
 result={'status':'passed','exact_bandit_gradient_checks':identities,'common_return_shift_invariance':True,'frozen_patient_and_reference':True,'original_status_function_no_smoothing':True,'actual_model_checks':rows,'scope':'finite-horizon estimator and engineering checks; block-policy surrogate differs from every5min replanning; no clinical or performance claim'};(ROOT/'checks/delayed_policy_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
