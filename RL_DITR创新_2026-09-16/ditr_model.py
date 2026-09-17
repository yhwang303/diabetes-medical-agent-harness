"""RL-DITR reference structure, adapted explicitly to continuous5-minute basal rates.
No ReBRAC actor/critic or fixed individual reference enters this model.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

def mlp(inputs,outputs,width=256):
 return nn.Sequential(nn.Linear(inputs,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,outputs))

class FixedScale(nn.Module):
 def __init__(self,scale):super().__init__();self.scale=scale
 def forward(self,x):return x*self.scale

def bounded_mlp(inputs,scale,width):
 return nn.Sequential(*list(mlp(inputs,1,width).children()),nn.Tanh(),FixedScale(scale))

class CategoricalScalar(nn.Module):
 """Author-style two-hot scalar support, expressed in current physical units."""
 def __init__(self,inputs,width,radius,bound):
  super().__init__();self.logits=mlp(inputs,2*radius+1,width);self.register_buffer('support',torch.linspace(-bound,bound,2*radius+1))
 def forward(self,x):
  logits=self.logits(x).float();return (logits.softmax(-1)*self.support).sum(-1,keepdim=True)
 def loss(self,logits,target):
  radius=(len(self.support)-1)//2;scaled=(target/self.support[-1]*radius).clamp(-radius,radius);lo=scaled.floor();fraction=scaled-lo
  distribution=torch.zeros_like(logits,dtype=torch.float32);i=(lo+radius).long();distribution.scatter_add_(-1,i.unsqueeze(-1),(1-fraction).unsqueeze(-1));distribution.scatter_add_(-1,(i+1).clamp_max(2*radius).unsqueeze(-1),fraction.unsqueeze(-1))
  # Match fixed author utils.logit_regression_loss: mean over bins, not sum.
  return -(distribution*logits.float().log_softmax(-1)).mean(-1)

def status_score(glucose):
 bg=(glucose*18).clamp_min(1)
 risk=10*(1.509*(torch.log(bg).pow(1.084)-5.381)).square()
 return torch.where(bg<70,-torch.ones_like(bg),1-risk.clamp(0,15.5)/7.75)

class PatientModel(nn.Module):
 def __init__(self,width=256,layers=3,heads=8,ff=2048,dropout=.4,bounded_heads=False,categorical_heads=False,transition_mode='recursive'):
  super().__init__();self.width=width;self.categorical_heads=categorical_heads;self.transition_mode=transition_mode
  if transition_mode not in ['recursive','action_prefix']:raise ValueError(transition_mode)
  t=torch.arange(512).float()[:,None];div=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000)/width));pos=torch.zeros(512,width);pos[:,0::2]=torch.sin(t*div);pos[:,1::2]=torch.cos(t*div);self.register_buffer('pos',pos)
  self.input=nn.Sequential(nn.Linear(22+width,width),nn.ReLU(),nn.LayerNorm(width))
  self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,ff,dropout,batch_first=False),layers,enable_nested_tensor=False)
  self.action=nn.Linear(1,width);self.time=nn.Linear(1,width)
  self.transition_input=nn.Sequential(nn.Linear(width*4,width),nn.ReLU(),nn.LayerNorm(width))
  self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,heads,ff,dropout,batch_first=False),layers)
  self.glucose=mlp(width,1,width);self.wtr=mlp(width,2,width);self.reward=mlp(width*2,1,width);self.value=mlp(width,1,width)
  if bounded_heads:
   # Original categorical reward/value supports are bounded. Preserve that property
   # in scalar continuous-time adaptation, using the known geometric return bound.
   self.reward=bounded_mlp(width*2,1.,width);self.value=bounded_mlp(width,(1/12)/(1-.9**(1/12)),width)
  if categorical_heads:
   self.reward=CategoricalScalar(width*2,width,1,1.);self.value=CategoricalScalar(width,width,20,(1/12)/(1-.9**(1/12)))
 def encode(self,history):
  b,l,_=history.shape;z=self.input(torch.cat([history,self.pos[:l].expand(b,-1,-1)],-1));mask=torch.ones(l,l,dtype=torch.bool,device=z.device).triu(1)
  return self.encoder(z.transpose(0,1),mask=mask).transpose(0,1)
 def step(self,z,action,memory,features=None):
  features=[] if features is None else list(features);h=len(features);a=self.action(action.reshape(-1,1)/5)
  dt=self.time(torch.full_like(action.reshape(-1,1),(h+1)/12));transition_state=memory[:,-1] if self.transition_mode=='action_prefix' else z
  features.append(torch.cat([transition_state,a,dt,self.pos[72+h].expand(len(z),-1)],-1));q=self.transition_input(torch.stack(features,1));l=q.shape[1]
  next_z=self.decoder(q.transpose(0,1),memory.transpose(0,1),tgt_mask=torch.ones(l,l,dtype=torch.bool,device=z.device).triu(1))[-1]
  reward=self.reward(torch.cat([z,a],-1)).squeeze(-1)/12
  return next_z,reward,features
 def rollout(self,history,actions,memory=None):
  memory=self.encode(history) if memory is None else memory;z=memory[:,-1];features=[];states=[];glucose=[];wtr=[];rewards=[]
  if self.transition_mode=='action_prefix':
   # All causal action prefixes are evaluated together; no predicted patient state
   # enters another transition token. This is the sole H1 structural intervention.
   batch,horizon=actions.shape;a=self.action(actions[:,:,None]/5);dt=self.time((torch.arange(1,horizon+1,device=z.device,dtype=z.dtype)/12)[None,:,None].expand(batch,-1,-1))
   q=self.transition_input(torch.cat([z[:,None].expand(-1,horizon,-1),a,dt,self.pos[72:72+horizon].expand(batch,-1,-1)],-1))
   latent=self.decoder(q.transpose(0,1),memory.transpose(0,1),tgt_mask=torch.ones(horizon,horizon,dtype=torch.bool,device=z.device).triu(1)).transpose(0,1)
   previous=torch.cat([z[:,None],latent[:,:-1]],1);reward=self.reward(torch.cat([previous,a],-1)).squeeze(-1)/12
   result={'glucose':self.glucose(latent).squeeze(-1),'wtr':self.wtr(latent),'states':latent,'reward':reward,'memory':memory,'initial_value':self.value(z).squeeze(-1)}
  else:
   for k in range(actions.shape[1]):
    z,r,features=self.step(z,actions[:,k],memory,features);states.append(z);glucose.append(self.glucose(z).squeeze(-1));wtr.append(self.wtr(z));rewards.append(r)
   result={'glucose':torch.stack(glucose,1),'wtr':torch.stack(wtr,1),'states':torch.stack(states,1),'reward':torch.stack(rewards,1),'memory':memory,'initial_value':self.value(memory[:,-1]).squeeze(-1)}
  if self.categorical_heads:
   previous=torch.cat([memory[:,-1,None],result['states'][:,:-1]],1);a=self.action(actions[:,:,None]/5)
   result['reward_logits']=self.reward.logits(torch.cat([previous,a],-1));result['initial_value_logits']=self.value.logits(memory[:,-1])
  return result

class ContinuousPolicy(nn.Module):
 """Censored Gaussian on[0,20]U/h: exact interior density and endpoint masses.
 Censoring affects generated actions, never rounds or edits logged target doses.
 """
 def __init__(self,width=256,initial_mean=1.):
  super().__init__();self.network=mlp(width,2,width);last=self.network[-1];nn.init.zeros_(last.weight)
  with torch.no_grad():last.bias.copy_(torch.tensor([math.log(initial_mean/(20-initial_mean)),math.log(.3)]))
 def parameters_at(self,z):
  raw=self.network(z).float();mean=20*torch.sigmoid(raw[:,0]);std=raw[:,1].clamp(-4,1).exp();return mean,std
 def sample(self,z):
  mean,std=self.parameters_at(z);return torch.normal(mean,std).clamp(0,20)
 def log_prob(self,z,action):
  mean,std=self.parameters_at(z);normal=torch.distributions.Normal(mean,std);interior=normal.log_prob(action)
  low=torch.special.log_ndtr((0-mean)/std);high=torch.special.log_ndtr((mean-20)/std)
  return torch.where(action<=0,low,torch.where(action>=20,high,interior))
 def candidates(self,z,count=10):
  mean,std=self.parameters_at(z);q=torch.linspace(.05,.95,count,device=z.device);noise=math.sqrt(2)*torch.erfinv(2*q-1)
  return (mean[:,None]+std[:,None]*noise).clamp(0,20)

class DITRAgent(nn.Module):
 def __init__(self,**cfg):
  super().__init__();response_config=cfg.pop('response_config',None);point_reward=cfg.pop('point_reward',False)
  if point_reward:
   assert response_config is None
   from point_reward_patient import PointRewardPatient
   self.patient=PointRewardPatient(**cfg)
  elif response_config is None:self.patient=PatientModel(**cfg)
  else:
   from response_patient import ResponsePatient
   self.patient=ResponsePatient(response_config,**cfg)
  self.policy=ContinuousPolicy(self.patient.width)
 def imagine(self,history,horizon=12,gamma=.9**(1/12)):
  # Patient is a fixed environment during policy learning. Actions detached for score-function gradients.
  with torch.no_grad():memory=self.patient.encode(history);z=memory[:,-1]
  features=[];rewards=[];logps=[];values=[]
  for k in range(horizon):
   action=self.policy.sample(z).detach();logps.append(self.policy.log_prob(z,action))
   with torch.no_grad():
    values.append(self.patient.value(z).squeeze(-1));z,reward,features=self.patient.step(z,action,memory,features);rewards.append(reward)
  with torch.no_grad():
   end=self.patient.value(z).squeeze(-1);returns=[]
   for r in reversed(rewards):end=r+gamma*end;returns.append(end)
  return torch.stack(logps,1),torch.stack(list(reversed(returns)),1),torch.stack(values,1)
 @torch.no_grad()
 def plan(self,history,horizon=12,beam_size=10,gamma=.9**(1/12)):
  assert len(history)==1,'planning one observed history at a time';memory=self.patient.encode(history);z=memory[:,-1];scores=z.new_zeros(1);features=[];paths=z.new_zeros((1,0));initial_values=[]
  for k in range(horizon):
   actions=self.policy.candidates(z,beam_size);branches=actions.shape[1];a=actions.reshape(-1);mem=memory.expand(len(z),-1,-1).repeat_interleave(branches,0);expanded_z=z.repeat_interleave(branches,0);fs=[f.repeat_interleave(branches,0) for f in features]
   next_z,reward,new_features=self.patient.step(expanded_z,a,mem,fs);cum=scores.repeat_interleave(branches)+gamma**k*reward;total=cum+gamma**(k+1)*self.patient.value(next_z).squeeze(-1)
   take=total.topk(min(beam_size,len(total))).indices;z=next_z[take];scores=cum[take];features=[f[take] for f in new_features];paths=torch.cat([paths.repeat_interleave(branches,0),a[:,None]],-1)[take]
   initial_values.append(float(total[take[0]]))
  return float(paths[0,0]),{'planned_actions_u_h':paths[0].tolist(),'plan_value':initial_values[-1],'horizon':horizon,'beam_size':beam_size}
 @torch.no_grad()
 def plan_batch(self,history,horizon=12,beam_size=10,gamma=.9**(1/12)):
  """Independent beams, batched only for GPU throughput; patients never share attention."""
  memory=self.patient.encode(history);batch=len(history);z=memory[:,-1];scores=z.new_zeros(batch,1);features=[];paths=z.new_zeros(batch,1,0)
  for k in range(horizon):
   beam=scores.shape[1];actions=self.policy.candidates(z,beam_size);branches=actions.shape[1];a=actions.reshape(-1)
   mem=memory[:,None].expand(-1,beam*branches,-1,-1).reshape(batch*beam*branches,memory.shape[1],memory.shape[2])
   fs=[f.repeat_interleave(branches,0) for f in features]
   nz,reward,nf=self.patient.step(z.repeat_interleave(branches,0),a,mem,fs)
   cumulative=scores.repeat_interleave(branches,1)+gamma**k*reward.reshape(batch,-1)
   total=cumulative+gamma**(k+1)*self.patient.value(nz).reshape(batch,-1)
   take=total.topk(min(beam_size,total.shape[1]),dim=1).indices
   flat=(torch.arange(batch,device=z.device)[:,None]*total.shape[1]+take).reshape(-1)
   z=nz[flat];scores=cumulative.gather(1,take);features=[f[flat] for f in nf]
   expanded_paths=torch.cat([paths.repeat_interleave(branches,1),a.reshape(batch,-1,1)],-1)
   paths=expanded_paths.gather(1,take[:,:,None].expand(-1,-1,k+1))
  return paths[:,0,0],{'planned_actions_u_h':paths[:,0].cpu().tolist(),'plan_value':total.gather(1,take)[:,0].cpu().tolist(),'horizon':horizon,'beam_size':beam_size}
