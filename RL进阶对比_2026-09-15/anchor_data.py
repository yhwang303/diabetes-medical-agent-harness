"""Causal patient reference from first eligible history; no target action or future row is used."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from rl_data import Replay,ROOT

class AnchorReplay(Replay):
 def __init__(self,split,device='cuda',verify=True):
  super().__init__(split,device,verify);p=ROOT/'anchor_data'/('%s.npz'%split);meta=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];self.anchor_manifest=meta
  with np.load(p) as f:anchors=torch.from_numpy(f['anchors_u_h']).to(device)
  self.anchor=(anchors[self.arrays['patient'].long()]/10-1).reshape(-1,1)
 def batch(self,indices):
  b=super().batch(indices);anchor=self.anchor[indices];b['state']=torch.cat([b['state'],anchor],-1);b['next_state']=torch.cat([b['next_state'],anchor],-1);return b

def prepare():
 import sys
 sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'));import pipeline as p
 dest=ROOT/'anchor_data';dest.mkdir(exist_ok=True)
 for split in ['train','validation']:
  ds=p.LoopDataset(split=split,mode='retrospective_multimodal');manifest=json.loads((ROOT/'data'/split/'manifest.json').read_text());anchors=[];details=[]
  for record in manifest['patients']:
   pid=record['id'];f,raw,allowed=ds._patient(pid);first=int(np.flatnonzero(allowed)[0]);v=f.basal_u_prev_5min.to_numpy(float)[first-71:first+1];assert np.isfinite(v).all();anchor=float(v.mean()*12);assert 0<=anchor<=20,(pid,anchor);anchors.append(anchor);details.append({'patient':pid,'first_origin':first,'latest_source_row':first,'history_rows':72,'reference_u_h':anchor})
  path=dest/(split+'.npz');np.savez(path,anchors_u_h=np.array(anchors,dtype='float32'));meta={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':split,'origin_count_unchanged':manifest['origins'],'patients':details,'source':'mean of actual prior-5min basal over first eligible 72-row history x12; fixed thereafter, all source rows <= earliest decision','normalization':'reference_Uh/10-1 appended to original 1584 flattened features','clinical_ready':False};path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(split,len(anchors),np.quantile(anchors,[0,.5,1]),flush=True)
if __name__=='__main__':prepare()
