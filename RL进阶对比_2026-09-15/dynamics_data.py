import json,hashlib
import numpy as np
import torch
from rl_data import Replay,ROOT

class Dynamics(Replay):
 def __init__(self,split,device='cuda'):
  super().__init__(split,device);folder=ROOT/'dynamics_data'/split;self.dynamics_manifest=json.loads((folder/'manifest.json').read_text())
  for name,sha in self.dynamics_manifest['array_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
  self.extra={p.stem:torch.from_numpy(np.load(p)).to(device) for p in folder.glob('*.npy')};self.normalizer=json.loads((ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text());self.mean=self.normalizer['cgm_mmol_l']['mean'];self.scale=self.normalizer['cgm_mmol_l']['scale']
 def batch(self,indices):
  rows=self.arrays['start'][indices];a=self.arrays;horizon=12;offsets=torch.arange(horizon,device=self.device);grid=(rows[:,None]+offsets).clamp(max=len(a['features'])-2);mask=offsets[None,:]<self.extra['length'][indices,None]
  state=torch.cat([a['features'][rows[:,None]+self.history_offsets],self.time.expand(len(rows),-1,-1)],-1)
  future=a['features'][grid+1];events=future[:,:,[2,7,3,8,4,9]];events=torch.where(mask[:,:,None],events,torch.zeros_like(events))
  target=torch.where(mask,(self.extra['glucose_grid'][grid+1]-self.mean)/self.scale,torch.zeros_like(grid,dtype=torch.float));actions=torch.where(mask,self.extra['action_grid'][grid],torch.zeros_like(grid,dtype=torch.float))
  return {'state':state,'actions':actions,'events':events,'target':target,'mask':mask}
