"""Task-adapted PyTorch ReBRAC/FQL; primary formulas and deviations in baseline_sources.md."""
import copy
import math
import torch
from torch import nn

class MLP(nn.Sequential):
 def __init__(self,inputs,width,layers,outputs=1,layer_norm=False,tanh=False,style='rebrac'):
  modules=[]
  for _ in range(layers):
   linear=nn.Linear(inputs,width)
   if style=='fql':nn.init.xavier_uniform_(linear.weight);nn.init.zeros_(linear.bias)
   else:nn.init.constant_(linear.bias,.1)
   modules.append(linear)
   modules.append(nn.GELU(approximate='tanh') if style=='fql' else nn.ReLU())
   if layer_norm:modules.append(nn.LayerNorm(width,eps=1e-6))
   inputs=width
  last=nn.Linear(inputs,outputs)
  if style=='fql':nn.init.xavier_uniform_(last.weight);nn.init.zeros_(last.bias)
  else:
   bound=.001 if tanh else .003;nn.init.uniform_(last.weight,-bound,bound);nn.init.uniform_(last.bias,-bound,bound)
  modules.append(last)
  if tanh:modules.append(nn.Tanh())
  super().__init__(*modules)

class AnchoredActor(nn.Module):
 def __init__(self,inputs,width,layers,style,residual_limit=None):
  super().__init__();self.network=MLP(inputs,width,layers,style=style);self.residual_limit=residual_limit
  if residual_limit is not None:assert residual_limit>0
  # Zero residual recovers the observed initial reference exactly.
  nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
 def forward(self,s):
  anchor=(s[:,1584:1585]+1)*10
  residual=2*self.network(s)
  if self.residual_limit is not None:residual=self.residual_limit*torch.tanh(residual/self.residual_limit)
  return (anchor+residual).clamp(0,20)/10-1

class TwinQ(nn.Module):
 def __init__(self,width,layers,style,state_dim=1584):
  super().__init__();self.q1=MLP(state_dim+1,width,layers,layer_norm=True,style=style);self.q2=MLP(state_dim+1,width,layers,layer_norm=True,style=style)
 def forward(self,s,a):
  x=torch.cat([s,a],-1);return torch.cat([self.q1(x),self.q2(x)],-1)

def masked_mean(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)

class Agent(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;self.algo=cfg['algorithm'];w,l=cfg['width'],cfg['layers']
  state_dim=cfg.get('state_dim',1584)
  self.actor=AnchoredActor(state_dim+(self.algo=='fql'),w,l,self.algo,cfg.get('residual_limit_u_h')) if cfg.get('anchor_context') else MLP(state_dim+(self.algo=='fql'),w,l,tanh=self.algo!='fql',style=self.algo)
  if 'initial_action_center' in cfg and not cfg.get('anchor_context'):
   center=cfg['initial_action_center'];last=next(m for m in reversed(self.actor) if isinstance(m,nn.Linear))
   nn.init.constant_(last.bias,center if self.algo=='fql' else math.atanh(center))
  self.q=TwinQ(w,l,self.algo,state_dim);self.q_target=copy.deepcopy(self.q).requires_grad_(False)
  if self.algo=='fql':
   self.flow=MLP(state_dim+2,w,l,style='fql')
   if 'initial_action_center' in cfg:nn.init.constant_(self.flow[-1].bias,cfg['initial_action_center'])
  else:self.actor_target=copy.deepcopy(self.actor).requires_grad_(False)
 def optimizers(self):
  c=self.cfg;self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=c['actor_lr']);self.q_opt=torch.optim.Adam(self.q.parameters(),lr=c['critic_lr'])
  if self.algo=='fql':self.flow_opt=torch.optim.Adam(self.flow.parameters(),lr=c['actor_lr'])
 def act(self,s,noise=None):
  if self.algo=='fql':
   if noise is None:noise=torch.randn((len(s),1),device=s.device)
   return self.actor(torch.cat([s,noise],-1)).clamp(-1,1)
  return self.actor(s)
 @torch.no_grad()
 def flow_action(self,s,z):
  x=z.clone();steps=self.cfg['flow_steps']
  for k in range(steps):x=x+self.flow(torch.cat([s,x,torch.full_like(x,k/steps)],-1))/steps
  return x.clamp(-1,1)
 @torch.no_grad()
 def target_update(self):
  tau=self.cfg['tau']
  for src,dst in zip(self.q.parameters(),self.q_target.parameters()):dst.lerp_(src,tau)
  if self.algo!='fql':
   for src,dst in zip(self.actor.parameters(),self.actor_target.parameters()):dst.lerp_(src,tau)
 def critic_loss(self,b):
  c=self.cfg;valid=b['q_valid'];s,a=b['state'],b['action']
  with torch.no_grad():
   if self.algo=='fql':
    next_a=self.act(b['next_state']);nq=self.q_target(b['next_state'],next_a)
    next_q=nq.mean(-1,keepdim=True) if c['q_agg']=='mean' else nq.min(-1,keepdim=True)[0]
   else:
    noise=(torch.randn_like(a)*c['target_noise']).clamp(-c['noise_clip'],c['noise_clip'])
    next_a=(self.actor_target(b['next_state'])+noise).clamp(-1,1)
    next_q=self.q_target(b['next_state'],next_a).min(-1,keepdim=True)[0]-c['beta_q']*(next_a-b['next_action']).square()
   target=b['reward']+c['gamma']*next_q
  q=self.q(s,a);errors=(q-target).square()
  per_row=errors.mean(-1,keepdim=True) if self.algo=='fql' else errors.sum(-1,keepdim=True)
  return masked_mean(per_row,valid),q,target
 def update(self,b,step):
  c=self.cfg;s,a=b['state'],b['action'];mask=b['q_valid'];info={}
  self.actor_opt.zero_grad(set_to_none=True);self.q_opt.zero_grad(set_to_none=True)
  if self.algo=='bc':
   actor_loss=(self.act(s)-a).square().mean();actor_loss.backward();self.actor_opt.step()
   return {'actor_loss':float(actor_loss),'bc_mse':float(actor_loss)}
  q_loss,q,target=self.critic_loss(b)
  if self.algo=='rebrac':
   q_loss.backward();self.q_opt.step()
   if step%c['policy_delay']==0:
    self.q.requires_grad_(False);pa=self.act(s);pqs=self.q(s,pa).min(-1,keepdim=True)[0]
    scale=masked_mean(pqs.abs(),mask).detach().clamp_min(1e-6).reciprocal()
    bc=(pa-a).square().mean();actor_loss=c['beta_actor']*bc-scale*masked_mean(pqs,mask)
    if hasattr(self,'model_guidance') and c['model_guidance']['weight']>0:
     model_loss,model_info=self.model_guidance(s[:,:1584],pa);actor_loss=actor_loss+c['model_guidance']['weight']*model_loss;info.update(model_info)
    actor_loss.backward();self.actor_opt.step();self.q.requires_grad_(True);self.target_update()
    info.update(actor_loss=float(actor_loss),bc_mse=float(bc))
  else:
   # All gradients use one common parameter snapshot, matching the source's joint loss update.
   self.flow_opt.zero_grad(set_to_none=True)
   z=torch.randn_like(a);t=torch.rand_like(a);xt=(1-t)*z+t*a
   flow_loss=(self.flow(torch.cat([s,xt,t],-1))-(a-z)).square().mean()
   noise=torch.randn_like(a);teacher=self.flow_action(s,noise)
   raw=self.actor(torch.cat([s,noise],-1));distill=(raw-teacher).square().mean()
   self.q.requires_grad_(False);pqs=self.q(s,raw.clamp(-1,1)).mean(-1,keepdim=True)
   q_term=-masked_mean(pqs,mask)
   if c.get('normalize_q_loss'):q_term=q_term*masked_mean(pqs.abs(),mask).detach().clamp_min(1e-6).reciprocal()
   actor_loss=c['alpha']*distill+q_term
   actor_loss.backward();self.q.requires_grad_(True);flow_loss.backward();q_loss.backward()
   self.actor_opt.step();self.flow_opt.step();self.q_opt.step();self.target_update()
   info.update(actor_loss=float(actor_loss),flow_loss=float(flow_loss),distill_loss=float(distill))
  assert torch.isfinite(q_loss) and torch.isfinite(q).all() and torch.isfinite(target).all()
  info.update(critic_loss=float(q_loss),q_mean=float(masked_mean(q.mean(-1,keepdim=True),mask)),target_mean=float(masked_mean(target,mask)),td_rows=int(mask.sum()))
  return info
