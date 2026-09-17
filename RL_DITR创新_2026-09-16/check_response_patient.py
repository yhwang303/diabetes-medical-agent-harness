"""Check learned response composition before policy optimization or control evaluation."""
import json
import torch
from ditr_data import ROOT,tensor
from ditr_model import DITRAgent,status_score
from ditr_losses import policy_loss
from paired_data import PairedData

def main():
 torch.set_num_threads(4);torch.manual_seed(260915);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 ck=torch.load(ROOT/'results/C01_response_patient_composition/patient_composed.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.patient.requires_grad_(False)
 data=PairedData();import numpy as np
 b=tensor(data.batch([0],np.random.default_rng(260915)));h=b['state'][:2];actions=b['action'][:2,:12]
 with torch.no_grad():
  memory=agent.patient.encode(h);reference=memory[:,0,0];repeated=reference[:,None].expand(-1,12);base=agent.patient.background.rollout(None,repeated,memory=memory[:,1:]);zero=agent.patient.rollout(h,repeated)
  torch.testing.assert_close(zero['glucose'],base['glucose'],rtol=0,atol=0);torch.testing.assert_close(zero['states'],base['states'],rtol=0,atol=0)
  full=agent.patient.rollout(h,actions);features=[];z=memory[:,-1];states=[];rewards=[]
  for k in range(12):z,r,features=agent.patient.step(z,actions[:,k],memory,features);states.append(z);rewards.append(r)
  torch.testing.assert_close(torch.stack(states,1),full['states'],rtol=2e-4,atol=2e-5);torch.testing.assert_close(torch.stack(rewards,1),full['reward'],rtol=2e-4,atol=2e-5)
  changed=actions.clone();changed[:,6:]+=1;later=agent.patient.rollout(h,changed);torch.testing.assert_close(later['glucose'][:,:6],full['glucose'][:,:6],rtol=0,atol=0);torch.testing.assert_close(later['states'][:,:6],full['states'][:,:6],rtol=0,atol=0)
  torch.testing.assert_close(full['reward'],status_score(full['glucose'])/12,rtol=0,atol=0)
  batched,_=agent.plan_batch(h,horizon=4,beam_size=3);separate=torch.tensor([agent.plan(h[i:i+1],horizon=4,beam_size=3)[0] for i in range(2)],device='cuda');torch.testing.assert_close(batched,separate,rtol=1e-4,atol=1e-5)
 loss,parts=policy_loss(agent,b,horizon=12,variant='bounded_return_nll');assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in agent.patient.parameters());grad=sum(float(p.grad.abs().sum()) for p in agent.policy.parameters() if p.grad is not None);assert grad>0 and np.isfinite(grad)
 result={'status':'passed','reference_units_u_h':reference.tolist(),'zero_intervention_exact_identity':True,'sequential_parallel_transition_parity':True,'future_action_causality':True,'reward_glucose_function_consistency':True,'batched_independent_planning_parity':True,'policy_gradient_l1':grad,'patient_frozen_during_policy_learning':True,'scope':'engineering/gradient checks; does not establish uncertainty calibration, nonlinear physiology, control efficacy or novelty'}
 (ROOT/'checks/response_patient_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
