"""RL-DITR patient transition composed from reference dynamics and learned interventions.
The encoder and reference decoder retain the original full-Loop training. Future
actions affect glucose AND the policy's latent state through causal response
operators trained on shared-state interventions. Reward uses the same status
function evaluated on predicted glucose; point-risk approximation is explicit.
"""
import torch
from torch import nn
from ditr_model import PatientModel,status_score
from response_operator import ResponseOperator,StateResponseOperator

class ResponsePatient(nn.Module):
 def __init__(self,response_config,**cfg):
  super().__init__();self.background=PatientModel(**cfg);self.width=self.background.width;self.categorical_heads=True;self.derived_reward=True
  self.glucose_response=ResponseOperator(conditioned=response_config.get('conditioned',True))
  self.state_response=StateResponseOperator()
  self.register_buffer('basal_mean',torch.tensor(response_config['basal_mean']))
  self.register_buffer('basal_scale',torch.tensor(response_config['basal_scale']))
 @property
 def value(self):return self.background.value
 @property
 def reward(self):return self.background.reward
 def encode(self,history):
  encoded=self.background.encode(history).float();observed=history[:,:,6]>.5;index=torch.arange(history.shape[1],device=history.device)[None].expand(len(history),-1);last=index.masked_fill(~observed,-1).max(1).values
  raw=history[torch.arange(len(history),device=history.device),last.clamp_min(0),1];reference=(raw*self.basal_scale+self.basal_mean)*12
  # Zero is only a coordinate origin when no basal is recorded; never a fallback dose.
  reference=torch.where(last>=0,reference,torch.zeros_like(reference));metadata=encoded.new_zeros(len(history),1,self.width);metadata[:,0,0]=reference;metadata[:,0,1]=(last>=0).to(encoded.dtype)
  return torch.cat([metadata,encoded],1)
 def rollout(self,history,actions,memory=None):
  memory=self.encode(history) if memory is None else memory;encoded=memory[:,1:];context=encoded[:,-1];reference=memory[:,0,0];reference_actions=reference[:,None].expand_as(actions)
  base=self.background.rollout(None,reference_actions,memory=encoded);difference=actions-reference_actions;dg=self.glucose_response(context,difference);dz=self.state_response(context,difference)
  glucose=base['glucose']+dg;states=base['states']+dz
  return {'glucose':glucose,'wtr':self.background.wtr(states),'states':states,'reward':status_score(glucose)/12,'memory':memory,'initial_value':self.value(context).squeeze(-1),'initial_value_logits':self.value.logits(context),'reference_glucose':base['glucose'],'glucose_response':dg,'reference_action_u_h':reference}
 def step(self,z,action,memory,features=None):
  features=[] if features is None else list(features);features.append(action.reshape(-1,1));actions=torch.cat(features,1);out=self.rollout(None,actions,memory=memory)
  return out['states'][:,-1],out['reward'][:,-1],features
