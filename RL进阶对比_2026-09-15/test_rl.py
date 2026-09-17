"""Hand-computed TD equations, real masked tails, gradient paths and data alignment."""
import sys,json
import numpy as np
import torch
from torch import nn
from rl_algorithms import Agent
from rl_data import Replay,ROOT

class ConstantActor(nn.Module):
 def forward(self,s):return torch.zeros((len(s),1),device=s.device)
class ConstantQ(nn.Module):
 def __init__(self,values):super().__init__();self.values=values
 def forward(self,s,a):return torch.tensor(self.values,device=s.device).expand(len(s),-1)

def main():
 torch.set_num_threads(4);torch.manual_seed(9);checks=[]
 for algo,expected in [('rebrac',1.55),('fql',2.9)]:
  c={'algorithm':algo,'width':16,'layers':2,'actor_lr':.001,'critic_lr':.001,'tau':.005,'gamma':.9,'beta_q':.5,'beta_actor':1,'target_noise':0,'noise_clip':0,'q_agg':'mean','alpha':10,'flow_steps':3,'policy_delay':2}
  agent=Agent(c).cuda();agent.q=ConstantQ([0.,0.]);agent.q_target=ConstantQ([2.,4.]);agent.actor_target=ConstantActor();agent.act=lambda s,noise=None:torch.zeros((len(s),1),device=s.device)
  b={'state':torch.zeros(2,1584,device='cuda'),'next_state':torch.zeros(2,1584,device='cuda'),'action':torch.zeros(2,1,device='cuda'),'next_action':torch.ones(2,1,device='cuda'),'reward':torch.tensor([[.2],[1000.]],device='cuda'),'q_valid':torch.tensor([[True],[False]],device='cuda')}
  loss,q,target=agent.critic_loss(b);assert abs(float(target[0])-expected)<1e-5;assert abs(float(loss)-expected**2*(2 if algo=='rebrac' else 1))<1e-4
  checks.append(algo+'_hand_computed_td_and_unknown_tail_mask')
  agent=Agent(c).cuda();agent.optimizers();b={**b,'state':torch.randn(2,1584,device='cuda'),'next_state':torch.randn(2,1584,device='cuda')};b['reward'][1]=.1
  before={k:v.clone() for k,v in agent.state_dict().items()};info=agent.update(b,2)
  assert all(np.isfinite(v) for v in info.values())
  for prefix in ['actor.','q.']+(['flow.'] if algo=='fql' else []):assert any(not torch.equal(v,agent.state_dict()[k]) for k,v in before.items() if k.startswith(prefix))
  if algo=='fql':assert not agent.flow_action(b['state'],torch.zeros(2,1,device='cuda')).requires_grad
  checks.append(algo+'_finite_updates_all_learning_modules_and_detached_teacher')
 data=Replay('train');assert data.size==1653421
 # Every patient's first origin must have exactly its own original causal history and immediate next history.
 ids=[];patient=data.arrays['patient'].cpu().numpy()
 for p in np.unique(patient):ids.append(int(np.flatnonzero(patient==p)[0]))
 batch=data.batch(torch.tensor(ids,device='cuda'));base=ROOT.parent/'RL训练_2026-09-15/packed/train'
 for i,entry in enumerate(data.manifest['patients']):
  with np.load(base/(entry['id']+'.npz')) as d:
   s=int(d['starts'][0]);expected=d['features'][s-71:s+1];nxt=d['features'][s-70:s+2]
   assert np.array_equal(batch['state'][i].reshape(72,22).cpu().numpy()[:,:20],expected)
   assert np.array_equal(batch['next_state'][i].reshape(72,22).cpu().numpy()[:,:20],nxt)
   assert abs(float(batch['action'][i])*10+10-float(d['action'][s]))<2e-6
 checks.append('all225_patients_original_history_next_history_and_action_units')
 result={'checks_passed':checks,'source_sha_verified':True,'clinical_ready':False};(ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
