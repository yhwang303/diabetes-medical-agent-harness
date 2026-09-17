"""CPU inference for exported policy: keeps simulator independent of GPU/Python Torch runtime."""
import json
from pathlib import Path
import numpy as np

class NumpyPolicy:
 def __init__(self,path,normalizer_path=None):
  path=Path(path);self.meta=json.loads(path.with_suffix('.json').read_text());self.cfg=self.meta['config']
  with np.load(path) as f:self.weights={k:f[k] for k in f.files}
  self.modules=self.meta['modules'];self.initial_reference=None
  if self.cfg.get('anchor_context'):
   norm_path=Path(normalizer_path) if normalizer_path is not None else Path(__file__).resolve().parent.parent/'Loop数据集/训练管线_v2/prepared/normalization.json'
   norm=json.loads(norm_path.read_text());self.basal_norm=norm['basal_u_prev_5min']
 def __call__(self,state,noise=None):
  x=np.asarray(state,dtype='float32')
  if self.cfg.get('anchor_context') and x.size==1584:
   history=x.reshape(72,22);observed=history[:,6]>0
   if self.initial_reference is None:self.initial_reference=float(np.mean((history[observed,1]*self.basal_norm['scale']+self.basal_norm['mean'])*12))
   x=np.concatenate([history.reshape(1,1584),np.array([[self.initial_reference/10-1]],dtype='float32')],-1)
  x=x.reshape(-1,self.cfg.get('state_dim',1584));anchor=(x[:,1584:1585]+1)*10 if self.cfg.get('anchor_context') else None
  if self.cfg['algorithm']=='fql':
   if noise is None:raise ValueError('FQL evaluation requires explicit seeded noise')
   x=np.concatenate([x,np.asarray(noise,dtype='float32').reshape(-1,1)],-1)
  for layer in self.modules:
   kind=layer['type'];key=layer['key']
   if kind=='Linear':x=x@self.weights[key+'.weight'].T+self.weights[key+'.bias']
   elif kind=='ReLU':x=np.maximum(x,0)
   elif kind=='GELU':x=.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x*x*x)))
   elif kind=='LayerNorm':
    mean=x.mean(-1,keepdims=True);var=((x-mean)**2).mean(-1,keepdims=True);x=(x-mean)/np.sqrt(var+layer['eps']);x=x*self.weights[key+'.weight']+self.weights[key+'.bias']
   elif kind=='Tanh':x=np.tanh(x)
   else:raise ValueError(kind)
  if self.meta.get('output_transform')=='anchor_plus_2_residual':
   residual=2*x
   if self.cfg.get('residual_limit_u_h') is not None:
    limit=self.cfg['residual_limit_u_h'];residual=limit*np.tanh(residual/limit)
   return np.clip(anchor+residual,0,20)/10-1
  return np.clip(x,-1,1)
