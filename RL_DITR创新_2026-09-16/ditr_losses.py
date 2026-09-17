"""Auditable RL-DITR losses and explicit continuous-time task adaptations."""
import torch
import torch.nn.functional as F
from ditr_model import status_score

GAMMA=.9**(1/12)

def masked_mean(x,mask):
 return (x*mask).sum()/mask.sum().clamp_min(1)

def discounted_return(reward,mask,bootstrap,gamma=GAMMA):
 """A missing future is not an absorbing terminal; bootstrap actual last state."""
 result=bootstrap
 for k in reversed(range(reward.shape[1])):
  result=torch.where(mask[:,k],reward[:,k]+gamma*result,result)
 return result

def patient_loss(model,target_model,b,mu=.1,return_rollout=False):
 out=model.rollout(b['state'],b['action']);mask=b['mask'];y=b['target']
 glu=masked_mean((out['glucose']-y).square(),mask)
 labels=((y*18>=70)&(y*18<=180)).long()
 wtr=masked_mean(F.cross_entropy(out['wtr'].transpose(1,2),labels,reduction='none'),mask)
 real_z=model.encode(b['next_state'])[:,-1]
 consistency=(out['states'][torch.arange(len(y),device=y.device),b['k']]-real_z).square().mean()
 status=status_score(y)
 # Train the unscaled status head; conversion to interval reward occurs in step().
 reward=masked_mean((out['reward']*12-status).square(),mask)
 with torch.no_grad():
  end=target_model.value(target_model.encode(b['final_state'])[:,-1]).squeeze(-1)
  if 'terminal' in b:end=end.masked_fill(b['terminal'],0)
  value_target=discounted_return(status/12,mask,end)
 value=(out['initial_value']-value_target).square().mean()
 reward_training=masked_mean(model.reward.loss(out['reward_logits'],status),mask) if model.categorical_heads else reward
 value_training=model.value.loss(out['initial_value_logits'],value_target).mean() if model.categorical_heads else value
 parts={'glucose_mse':glu,'wtr_ce':wtr,'consistency':consistency,'reward_status_mse':reward,'value_mse':value,'reward_training_loss':reward_training,'value_training_loss':value_training}
 loss=glu+wtr+mu*consistency+reward_training+value_training
 return (loss,parts,out) if return_rollout else (loss,parts)

def policy_loss(agent,b,horizon=12,epsilon_model=1.,epsilon_sl=1.,variant='legacy'):
 # f_R/f_T/reward/V frozen: the policy cannot change its evaluation environment.
 with torch.no_grad():
  z=agent.patient.encode(b['state'])[:,-1]
  end=agent.patient.value(agent.patient.encode(b['final_state'])[:,-1]).squeeze(-1)
  historical_return=discounted_return(status_score(b['target'])/12,b['mask'],end)
 historical_logp=agent.policy.log_prob(z,b['action'][:,0])
 historical=-(historical_return*historical_logp).mean()
 logps,returns,values=agent.imagine(b['state'],horizon)
 imagined=-(returns*logps).mean()
 mean,std=agent.policy.parameters_at(z)
 supervised=(mean-b['action'][:,0]).square().mean()
 action_mse=supervised
 if variant=='bounded_return_nll':
  if not agent.patient.categorical_heads or epsilon_sl<1:raise ValueError('Bounded categorical value and SL coefficient>=1 required')
  bound=(1/12)/(1-GAMMA)
  # Proper continuous analogue of supervised CE. Combined historical+SL
  # coefficient is epsilon_sl + R/bound >= 0, including negative returns.
  normalized=(historical_return/bound).clamp(-1,1)
  historical=-(normalized*historical_logp).mean();supervised=-historical_logp.mean()
  # A state-only baseline changes variance, not the expected score-function gradient.
  imagined=-(((returns-values)/bound).detach()*logps).mean()
 elif variant!='legacy':raise ValueError(variant)
 parts={'historical_policy_loss':historical,'imagined_policy_loss':imagined,'supervised_action_mse':action_mse,'supervised_training_loss':supervised,'mean_std':std.mean(),'imagined_return':returns.mean(),'historical_return_min':historical_return.min(),'historical_negative_fraction':(historical_return<0).float().mean()}
 if variant=='bounded_return_nll':parts['combined_history_weight_min']=epsilon_sl+normalized.min()
 return historical+epsilon_model*imagined+epsilon_sl*supervised,parts
