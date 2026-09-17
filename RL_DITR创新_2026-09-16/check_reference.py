"""Numerical/reference-chain checks on the actual CUDA environment and data."""
import itertools,json,math,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ditr_model import DITRAgent,ContinuousPolicy,status_score
from ditr_losses import discounted_return,patient_loss,policy_loss
from ditr_data import Data,tensor,ROOT

def main():
 torch.set_num_threads(8);torch.manual_seed(260915);torch.backends.cuda.matmul.allow_tf32=True
 checks={}
 reward=torch.tensor([[1.,2.,99.],[3.,99.,99.]])
 mask=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
 result=discounted_return(reward,mask,torch.tensor([10.,20.]),.5)
 assert torch.allclose(result,torch.tensor([4.5,13.]));checks['variable_length_bootstrap_not_terminal']=True
 assert torch.allclose(status_score(torch.tensor([50./18])),torch.tensor([-1.]));checks['reward_low_glucose']=True
 p=ContinuousPolicy(8);z=torch.zeros(3,8);a=torch.tensor([0.,1.,20.]);logp=p.log_prob(z,a)
 assert torch.isfinite(logp).all();(-logp.mean()).backward();assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
 expected=torch.distributions.Normal(torch.tensor(1.),torch.tensor(.3)).cdf(torch.tensor(0.)).log()
 assert abs(float(logp[0]-expected))<.001;checks['censored_policy_endpoint_masses_and_gradients']=True
 class Toy(nn.Module):
  def encode(self,h):return torch.zeros(len(h),72,1)
  def step(self,z,a,memory,features=None):return z+a[:,None],-(a-z[:,0]).square(),list(features or [])+[z]
  def value(self,z):return .2*z
 class Candidates(nn.Module):
  def candidates(self,z,count):return torch.tensor([.2,1.2]).expand(len(z),-1)
 toy=DITRAgent(width=8,layers=1,heads=2,ff=16,dropout=0);toy.patient=Toy();toy.policy=Candidates()
 action,info=toy.plan(torch.zeros(1,72,22),horizon=2,beam_size=2,gamma=.9)
 exact=[]
 for a1,a2 in itertools.product([.2,1.2],repeat=2):exact.append((-a1*a1+.9*-(a2-a1)**2+.9**2*.2*(a1+a2),a1,a2))
 best=max(exact);assert abs(info['plan_value']-best[0])<1e-6 and abs(action-best[1])<1e-6
 checks['beam_cumulative_reward_terminal_value_matches_exhaustive']=True
 data=Data('train');assert data.total==1653421 and len(data.patients)==225
 raw=next(data.batches(32,260915));b=tensor(raw)
 d=data.patients[raw['patient']];np.testing.assert_array_equal(raw['final_state'],data.history(d,raw['row']+raw['length']))
 checks['frozen_full_data_and_actual_bootstrap_state']=True
 agent=DITRAgent().cuda();target=copy.deepcopy(agent.patient).eval();target.requires_grad_(False)
 optimizer=torch.optim.Adam(agent.patient.parameters(),lr=1e-3,weight_decay=1e-4)
 timings=[];losses=[]
 for i in range(3):
  start=time.time();agent.patient.train();optimizer.zero_grad(set_to_none=True)
  loss,parts=patient_loss(agent.patient,target,b)
  assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(agent.patient.parameters(),5.)
  assert torch.isfinite(grad);optimizer.step();torch.cuda.synchronize()
  timings.append(time.time()-start);losses.append({k:float(v) for k,v in parts.items()})
 agent.patient.eval();agent.patient.requires_grad_(False);agent.zero_grad(set_to_none=True)
 loss,parts=policy_loss(agent,b);assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is None for p in agent.patient.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.policy.parameters())
 assert all(torch.isfinite(p.grad).all() for p in agent.policy.parameters() if p.grad is not None)
 checks['actual_batch_patient_backward_and_frozen_policy_training']=True
 agent.eval();start=time.time();a,info=agent.plan(b['state'][:1]);torch.cuda.synchronize();elapsed=time.time()-start
 assert 0<=a<=20 and len(info['planned_actions_u_h'])==12 and math.isfinite(info['plan_value'])
 checks['actual_model_beam10_horizon12']=True
 result={'checks':checks,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':260915,'patient_parameters':sum(p.numel() for p in agent.patient.parameters()),'patient_batch32_fp32_step_seconds':timings,'smoke_losses_not_formal_training':losses,'policy_smoke_parts':{k:float(v) for k,v in parts.items()},'beam_seconds':elapsed,'beam_output':info,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'complete_reference_training':False}
 (ROOT/'checks').mkdir(exist_ok=True);(ROOT/'checks/reference_smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
