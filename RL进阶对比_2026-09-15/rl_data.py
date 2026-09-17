from pathlib import Path
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

class Replay:
 def __init__(self,split,device='cuda',verify=True):
  folder=ROOT/'data'/split;self.manifest=json.loads((folder/'manifest.json').read_text());self.device=device
  if verify:
   for file,digest in self.manifest['array_sha256'].items():
    h=hashlib.sha256()
    with (folder/file).open('rb') as f:
     for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==digest,file
  self.arrays={p.stem:torch.from_numpy(np.load(p)).to(device) for p in folder.glob('*.npy')}
  self.size=len(self.arrays['start']);self.history_offsets=torch.arange(-71,1,device=device)
  self.time=torch.stack([torch.ones(72,device=device)/12,self.history_offsets/12],-1).float()
 def batch(self,indices):
  rows=self.arrays['start'][indices];a=self.arrays
  state=torch.cat([a['features'][rows[:,None]+self.history_offsets],self.time.expand(len(rows),-1,-1)],-1).flatten(1)
  next_state=torch.cat([a['features'][rows[:,None]+1+self.history_offsets],self.time.expand(len(rows),-1,-1)],-1).flatten(1)
  return {'state':state,'next_state':next_state,**{k:a[k][indices].reshape(-1,1) for k in ['action','next_action','reward','q_valid','bootstrap_valid','next_action_available']}}
