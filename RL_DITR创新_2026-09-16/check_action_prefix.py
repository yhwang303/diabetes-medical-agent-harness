"""Tests for the H1 computational dependency, not performance claims."""
import copy,json,time
import numpy as np
import torch
from ditr_model import DITRAgent
from ditr_data import Data,tensor,ROOT
from ditr_losses import patient_loss,policy_loss

torch.set_num_threads(4);torch.manual_seed(260915);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
data=Data('train');raw=next(data.batches(8,260915));b=tensor(raw)
agent=DITRAgent(categorical_heads=True,transition_mode='action_prefix').cuda().eval();model=agent.patient
with torch.no_grad():
 memory=model.encode(b['state']);parallel=model.rollout(b['state'],b['action'],memory);z=memory[:,-1];features=[];states=[];rewards=[]
 for k in range(12):z,r,features=model.step(z,b['action'][:,k],memory,features);states.append(z);rewards.append(r)
 torch.testing.assert_close(torch.stack(states,1),parallel['states'],atol=2e-5,rtol=2e-5);torch.testing.assert_close(torch.stack(rewards,1),parallel['reward'],atol=2e-5,rtol=2e-5)
 modified=b['action'].clone();modified[:,6:]+=1;other=model.rollout(b['state'],modified,memory);torch.testing.assert_close(other['states'][:,:6],parallel['states'][:,:6],atol=1e-6,rtol=1e-6)
 z=memory[:,-1];a1,_,_=model.step(z,b['action'][:,0],memory);a2,_,_=model.step(z+10,b['action'][:,0],memory);torch.testing.assert_close(a1,a2,atol=0,rtol=0)
target=copy.deepcopy(model).eval();target.requires_grad_(False);model.train();start=time.time();loss,parts=patient_loss(model,target,b);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
agent.zero_grad(set_to_none=True);model.eval().requires_grad_(False);ploss,pparts=policy_loss(agent,b);ploss.backward();assert all(p.grad is None for p in model.parameters());assert any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.policy.parameters())
result={'parallel_vs_sequential_equivalent':True,'future_action_does_not_leak_into_earlier_predictions':True,'transition_output_independent_of_previous_predicted_patient_latent':True,'finite_real_batch_patient_and_policy_gradients':True,'patient_frozen_during_policy':True,'parameter_count':sum(p.numel() for p in model.parameters()),'wall_seconds':time.time()-start,'note':'dependency and gradient tests only, no efficacy or novelty conclusion'}
(ROOT/'checks/action_prefix.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
