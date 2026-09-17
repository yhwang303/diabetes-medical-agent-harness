"""Finite delayed-return policy improvement with a frozen RL-DITR patient.

Known estimators, not claimed as new: independent leave-one-out REINFORCE and
trajectory KL to the inherited policy. Sampling gradients preserve the original
discontinuous low-glucose status cost; no pathwise derivative through that cost.
"""
import torch

def leave_one_out_advantage(returns):
 if returns.ndim!=2 or returns.shape[1]<2:raise ValueError('Independent trajectories per observed history required')
 return returns-(returns.sum(1,keepdim=True)-returns)/(returns.shape[1]-1)

def delayed_trajectory_loss(agent,reference_policy,history,horizon=48,block_steps=16,trajectories=4,kl_beta=1.,return_details=False):
 if horizon%block_steps or trajectories<2:raise ValueError('Invalid trajectory contract')
 batch=len(history);gamma=.9**(1/12)
 with torch.no_grad():
  memory=agent.patient.encode(history).repeat_interleave(trajectories,0);z=memory[:,-1]
 plans=[];logps=[];reference_logps=[]
 for k in range(horizon//block_steps):
  # Separate torch.normal draws are independent, not antithetic paired samples.
  # Match DITRAgent.imagine: detach the draw, but perform the first actor forward
  # with gradients enabled. Torch2.0 autocast otherwise caches no-grad weight
  # casts from sampling and reuses them for the differentiable log-probability.
  action=agent.policy.sample(z).detach()
  logps.append(agent.policy.log_prob(z,action))
  with torch.no_grad():
   reference_logps.append(reference_policy.log_prob(z,action));plans.append(action[:,None].expand(-1,block_steps));actions=torch.cat(plans,1)
   out=agent.patient.rollout(None,actions,memory=memory);z=out['states'][:,-1]
 logp=torch.stack(logps,1).sum(1).reshape(batch,trajectories)
 with torch.no_grad():
  reference_logp=torch.stack(reference_logps,1).sum(1).reshape(batch,trajectories)
  discount=gamma**torch.arange(horizon,device=history.device);status_return=(out['reward']*discount).sum(1).reshape(batch,trajectories)
  sampled_kl=logp.detach()-reference_logp;shaped_return=status_return-kl_beta*sampled_kl
  advantage=leave_one_out_advantage(shaped_return)
 loss=-(advantage.detach()*logp).mean()
 parts={'delayed_policy_loss':loss,'imagined_return':status_return.mean(),'within_history_return_std':status_return.std(1).mean(),'trajectory_sampled_kl':sampled_kl.mean(),'trajectory_sampled_kl_abs':sampled_kl.abs().mean(),'loo_advantage_rms':advantage.square().mean().sqrt(),'action_mean':actions.mean(),'action_zero_fraction':(actions==0).float().mean(),'action_upper_fraction':(actions==20).float().mean()}
 if return_details:return loss,parts,{'actions':actions,'logp':logp,'advantage':advantage,'sampled_kl':sampled_kl,'status_return':status_return}
 return loss,parts
