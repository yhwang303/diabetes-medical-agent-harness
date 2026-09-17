"""One-step RL arrays from frozen Loop v2. Retain every actor origin; mask unknown TD tails."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline as p

def main():
 out=ROOT/'data';out.mkdir(exist_ok=True)
 for split in ['train','validation']:
  ds=p.LoopDataset(split=split,mode='retrospective_multimodal')
  old=ROOT.parent/'RL训练_2026-09-15/packed'/split
  manifest=json.loads((old/'manifest.json').read_text());all_features=[];parts=[];offset=0;patients=[]
  for number,record in enumerate(manifest['patients']):
   pid=record['patient_id'];f,raw,allowed=ds._patient(pid)
   with np.load(old/(pid+'.npz')) as z:features=z['features'];starts=z['starts'].astype('int64')
   assert np.array_equal(starts,np.flatnonzero(allowed))
   nxt=starts+1;current=f.basal_action_u_h.to_numpy(float)[starts]
   assert np.isfinite(current).all() and (current>=0).all() and (current<=20).all()
   bootstrap=(f.cgm_mask & f.basal_bolus_history_6h_ok & f.cgm_history_6h_gap_le1h & f.in_study_window).to_numpy(bool)[nxt]
   next_available=allowed[nxt] & (f.episode_id.to_numpy()[nxt]==f.episode_id.to_numpy()[starts])
   next_action=f.basal_action_u_h.to_numpy(float)[nxt]
   assert np.isfinite(next_action[next_available]).all()
   glu=f.cgm_mmol_l.to_numpy(float)[nxt];reward=p.paper_status_score(glu)/12
   q_valid=bootstrap & next_available
   # Placeholder next_action is always accompanied by a validity mask; no TD use when unknown.
   parts.append({'start':starts+offset,'action':current/10-1,'next_action':np.where(next_available,next_action/10-1,0),'reward':reward,'q_valid':q_valid,'bootstrap_valid':bootstrap,'next_action_available':next_available,'patient':np.full(len(starts),number,dtype='int32')})
   all_features.append(features);offset+=len(features)
   patients.append({'id':pid,'origins':len(starts),'q_valid':int(q_valid.sum()),'bootstrap_valid':int(bootstrap.sum()),'next_action_available':int(next_available.sum())})
   if number%25==0:print(split,number,flush=True)
  dest=out/split;dest.mkdir(exist_ok=True)
  arrays={'features':np.concatenate(all_features)}
  arrays.update({k:np.concatenate([x[k] for x in parts]) for k in parts[0]})
  hashes={}
  for name,a in arrays.items():
   if a.dtype.kind=='f':a=a.astype('float32')
   path=dest/(name+'.npy');np.save(path,a)
   h=hashlib.sha256()
   with path.open('rb') as file:
    for chunk in iter(lambda:file.read(8*1024*1024),b''):h.update(chunk)
   hashes[path.name]=h.hexdigest()
  result={'split':split,'patients':patients,'patient_count':len(patients),'origins':sum(x['origins'] for x in patients),'q_valid':sum(x['q_valid'] for x in patients),'state_shape':[72,22],'feature_presence_filter':False,'action_mapping':'a_norm=a_Uh/10-1; exact inverse 10*(a_norm+1); no original action clipping','gamma':float(.9**(1/12)),'reward':'frozen v2 paper_status_score(next_cgm)/12','td_scope':'common valid-bootstrap and observed-next-action intersection; all origins retained for behavior losses; no fabricated terminal or next action','source_manifest_sha256':hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest(),'normalizer_sha256':p.sha(p.ROOT/'prepared/normalization.json'),'array_sha256':hashes,'clinical_ready':False}
  assert result['origins']==manifest['sample_count']
  (dest/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['patients','array_sha256']},indent=2),flush=True)

if __name__=='__main__':main()
