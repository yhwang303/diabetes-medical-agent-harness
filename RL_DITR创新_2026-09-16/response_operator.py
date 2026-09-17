"""State-conditioned causal response operator; no sign or dose-response oracle embedded.

Local assumption: within one four-hour query, the effect relative to a reference
plan is linear and time-invariant conditional on observed history. This is a
testable structural restriction, not a property guaranteed for real physiology.
"""
import math
import torch
from torch import nn

def response_basis(horizon=48,degree=7):
 t=torch.linspace(0,1,horizon)
 return torch.stack([math.comb(degree,j)*t.pow(j)*(1-t).pow(degree-j) for j in range(degree+1)],-1)

class ResponseOperator(nn.Module):
 def __init__(self,width=256,hidden=64,degree=7,horizon=48,conditioned=True):
  super().__init__();self.conditioned=conditioned;self.horizon=horizon
  self.register_buffer('basis',response_basis(horizon,degree))
  if conditioned:
   self.coefficients=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),nn.Tanh(),nn.Linear(hidden,degree+1))
   nn.init.zeros_(self.coefficients[-1].weight);nn.init.zeros_(self.coefficients[-1].bias)
  else:self.coefficients=nn.Parameter(torch.zeros(degree+1))
 def design(self,action_difference):
  """Actions in U/h; output basis sums, preserving every causal time lag."""
  steps=action_difference.shape[-1]
  if steps>self.horizon:raise ValueError('Beyond learned response horizon')
  index=torch.arange(steps,device=action_difference.device);lag=index[:,None]-index[None,:]
  kernel=self.basis[lag.clamp_min(0)]*(lag>=0).unsqueeze(-1)
  return torch.einsum('...s,tsk->...tk',action_difference,kernel)
 def forward(self,context,action_difference):
  coefficients=self.coefficients(context) if self.conditioned else self.coefficients.expand(len(context),-1)
  design=self.design(action_difference)
  while coefficients.ndim<design.ndim:coefficients=coefficients.unsqueeze(1)
  return (design*coefficients).sum(-1)

class StateResponseOperator(nn.Module):
 """Low-rank latent response, learned against differences of observed encodings."""
 def __init__(self,width=256,hidden=128,degree=7,rank=16,horizon=48):
  super().__init__();self.rank=rank;self.degree=degree
  self.design_operator=ResponseOperator(conditioned=False,degree=degree,horizon=horizon)
  self.design_operator.coefficients.requires_grad_(False)
  self.coefficients=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),nn.Tanh(),nn.Linear(hidden,(degree+1)*rank))
  self.projection=nn.Linear(rank,width,bias=False)
  nn.init.zeros_(self.coefficients[-1].weight);nn.init.zeros_(self.coefficients[-1].bias)
 def forward(self,context,action_difference):
  coefficients=self.coefficients(context).reshape(len(context),self.degree+1,self.rank)
  design=self.design_operator.design(action_difference)
  if action_difference.ndim==2:latent=torch.einsum('btk,bkr->btr',design,coefficients)
  elif action_difference.ndim==3:latent=torch.einsum('batk,bkr->batr',design,coefficients)
  else:raise ValueError('Expected one plan or multiple arms per history')
  return self.projection(latent)

class DirectPrefixResponse(nn.Module):
 """Near-parameter-matched causal MLP; no convolution/time-invariance restriction."""
 def __init__(self,width=256,hidden=54,degree=7,horizon=48):
  super().__init__();self.horizon=horizon;self.norm=nn.LayerNorm(width)
  self.register_buffer('basis',response_basis(horizon,degree))
  self.network=nn.Sequential(nn.Linear(width+horizon+degree+1,hidden),nn.Tanh(),nn.Linear(hidden,1))
  nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
 def forward(self,context,action_difference):
  single=action_difference.ndim==2
  if single:action_difference=action_difference[:,None]
  batch,arms,steps=action_difference.shape
  if steps>self.horizon:raise ValueError('Beyond learned response horizon')
  padded=torch.nn.functional.pad(action_difference,(0,self.horizon-steps))
  causal=torch.arange(self.horizon,device=context.device)[None]<=torch.arange(steps,device=context.device)[:,None]
  prefix=padded[:,:,None]*causal[None,None]
  z=self.norm(context)[:,None,None].expand(-1,arms,steps,-1)
  t=self.basis[:steps][None,None].expand(batch,arms,-1,-1)
  result=(self.network(torch.cat([z,prefix,t],-1))-self.network(torch.cat([z,torch.zeros_like(prefix),t],-1))).squeeze(-1)
  return result[:,0] if single else result
