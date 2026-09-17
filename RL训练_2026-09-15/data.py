from pathlib import Path
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
class Data:
 def __init__(self,split):
  self.root=ROOT/'packed'/split;self.manifest=json.loads((self.root/'manifest.json').read_text());self.patients={};self.split=split
  for p in self.manifest['patients']:
   with np.load(self.root/(p['patient_id']+'.npz')) as f: self.patients[p['patient_id']]={k:f[k] for k in f.files}
  self.total=self.manifest['sample_count']
 def history(self,d,rows):
  x=d['features'][rows[:,None]+np.arange(-71,1)]
  t=np.broadcast_to(np.stack([np.ones(72)/12,np.arange(-71,1)/12],-1),(len(rows),72,2))
  return np.concatenate([x,t],-1).astype('float32')
 def batch(self,pid,positions,horizon=12,rng=None):
  d=self.patients[pid];s=d['starts'][positions].astype('int64');length=np.minimum(d['length'][positions],horizon)
  row=np.minimum(s[:,None]+np.arange(horizon),len(d['features'])-2)
  mask=np.arange(horizon)[None,:]<length[:,None]
  actions=np.where(mask,d['action'][row],0).astype('float32')
  glu=np.where(mask,d['glucose'][row+1],0).astype('float32')
  assert np.isfinite(actions).all() and np.isfinite(glu).all()
  # A uniformly sampled valid k gives unbiased per-trajectory consistency, both sides differentiable.
  k=(rng.integers(0,2**31,size=len(s))%length+1) if rng is not None else length
  return {'state':self.history(d,s),'action':actions,'target':glu,'mask':mask,'next_state':self.history(d,s+k),'k':k.astype('int64')-1,'row':s,'length':length,'patient':pid}
 def batches(self,batch_size,seed,horizon=12,shuffle=True):
  rng=np.random.default_rng(seed);groups={pid:rng.permutation(len(d['starts'])) if shuffle else np.arange(len(d['starts'])) for pid,d in self.patients.items()};names=list(groups)
  # Round-robin patients: finite budgets still visit every patient, all rows exactly once per epoch.
  if shuffle:rng.shuffle(names)
  for offset in range(0,max(map(len,groups.values())),batch_size):
   for pid in names:
    positions=groups[pid][offset:offset+batch_size]
    if len(positions):yield self.batch(pid,positions,horizon,rng)

def tensor(batch,device='cuda'):
 return {k:torch.from_numpy(v).to(device) if isinstance(v,np.ndarray) else v for k,v in batch.items()}
