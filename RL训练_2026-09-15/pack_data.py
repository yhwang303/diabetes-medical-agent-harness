"""Compact causal arrays, preserving all eligible 5-min transitions and original patient split."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline as p

def pack(split):
 ds=p.LoopDataset(split=split,mode='retrospective_multimodal')
 out=ROOT/'packed'/split; out.mkdir(parents=True,exist_ok=True)
 parts=[]; offset=0; records=[]
 # Keep every source grid row; window indices never cross patient or episode.
 for number,(pid,idx) in enumerate(ds.index.groupby('patient_id')):
  f,raw,allowed=ds._patient(pid); n=len(f)
  val,obs,age,known=raw
  mu=np.array([ds.normalizer[k]['mean'] for k in p.CORE+p.AUX]); sd=np.array([ds.normalizer[k]['scale'] for k in p.CORE+p.AUX])
  features=np.concatenate([np.where(obs,(val-mu)/sd,0),obs,age,known],1).astype('float32')
  starts=idx.row_index.to_numpy(); length=np.zeros(len(starts),np.int16)
  valid=np.ones(len(starts),bool); ep=f.episode_id.to_numpy()
  for h in range(12):
   row=np.minimum(starts+h,n-1); valid &= (starts+h<n-1)&allowed[row]&(ep[row]==ep[starts]); length+=valid
  assert (length>=1).all()
  arrays={'features':features,'glucose':f.cgm_mmol_l.to_numpy('float32'),'action':f.basal_action_u_h.to_numpy('float32'),'starts':starts.astype('int32'),'length':length,'bolus':f.bolus_recorded_u_next_5min.to_numpy('float32')}
  np.savez(out/(pid+'.npz'),**arrays)
  records.append({'patient_id':pid,'rows':n,'samples':len(starts),'horizon30_samples':int((length>=6).sum()),'horizon60_samples':int((length>=12).sum())})
  if number%25==0: print(split,number,flush=True)
 manifest={'split':split,'patient_count':len(records),'sample_count':sum(x['samples'] for x in records),'patients':records,'contract_sha256':p.sha(p.ROOT/'contract.json'),'normalizer_sha256':p.sha(p.ROOT/'prepared/normalization.json'),'source_audit_sha256':p.sha(p.ROOT/'prepared/audit.json'),'mode':'retrospective_multimodal','presence_filter':False}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps({k:v for k,v in manifest.items() if k!='patients'}),flush=True)
if __name__=='__main__':
 for split in ['train','validation']: pack(split)
