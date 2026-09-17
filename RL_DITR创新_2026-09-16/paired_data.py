"""Training-only common-state intervention groups; no evaluation samples loaded."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from ditr_data import ROOT,tensor
from ditr_losses import masked_mean
from ditr_model import status_score

class PairedData:
 def __init__(self,folder='paired_sim_train'):
  assert folder in ['paired_sim_train','paired_sim_train_temporal']
  arms=5 if folder=='paired_sim_train' else 7
  root=ROOT/folder;self.manifest=json.loads((root/'manifest.json').read_text());assert self.manifest['status']=='complete'
  assert self.manifest['contract']['scenario_seeds']==[40101,40102]
  chunks=[]
  for item in self.manifest['scenarios']:
   path=root/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
   if not item['groups']:continue
   with np.load(path) as f:chunks.append({k:f[k] for k in f.files})
  self.arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};self.total=len(self.arrays['features'])
  assert self.total==self.manifest['groups'];assert self.arrays['features'].shape[1:]==(arms,120,20)
  x=self.arrays['features'];np.testing.assert_array_equal(x[:,:,:72],np.repeat(x[:,0:1,:72],arms,axis=1))
  assert np.all(self.arrays['length']>0);assert np.isfinite(x).all()
  assert np.array_equal(self.arrays['mask'],np.arange(48)[None,None,:]<self.arrays['length'][:,:,None])
  self.clock=np.stack([np.ones(72)/12,np.arange(-71,1)/12],-1).astype('float32')
 def batch(self,indices,rng):
  a={k:v[indices].reshape((-1,)+v.shape[2:]) for k,v in self.arrays.items() if k!='minute'};n=len(a['features']);length=a['length'].astype('int64');k=rng.integers(0,2**31,size=n)%length+1
  def history(end):
   features=a['features'][np.arange(n)[:,None],end[:,None]+np.arange(-71,1)]
   return np.concatenate([features,np.broadcast_to(self.clock,(n,72,2))],-1).astype('float32')
  return {'state':history(np.full(n,71)),'action':a['action'],'target':a['target'],'mask':a['mask'],'next_state':history(71+k),'final_state':history(71+length),'k':k-1,'length':length,'terminal':a['terminal'],'groups':len(indices)}
 def batches(self,groups,seed):
  rng=np.random.default_rng(seed)
  while True:
   indices=rng.permutation(self.total)
   for offset in range(0,self.total,groups):yield self.batch(indices[offset:offset+groups],rng)

def paired_path_loss(prediction,b):
 """Differences cancel shared future disturbances; no assumed monotone response."""
 count=b['groups'];mask=b['mask'].reshape(count,5,-1);joint=mask[:,1:]&mask[:,0:1]
 y=b['target'].reshape(count,5,-1);g=prediction['glucose'].reshape(count,5,-1)
 dg=(g[:,1:]-g[:,0:1])-(y[:,1:]-y[:,0:1]);glucose=masked_mean(dg.square(),joint)
 r=(prediction['reward']*12).reshape(count,5,-1);target=status_score(y)
 dr=(r[:,1:]-r[:,0:1])-(target[:,1:]-target[:,0:1]);risk=masked_mean(dr.square(),joint)
 return glucose+risk,{'paired_glucose_effect_mse':glucose,'paired_status_effect_mse':risk}
