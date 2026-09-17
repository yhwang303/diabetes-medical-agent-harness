from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline as p
out=ROOT/'results'; out.mkdir(exist_ok=True)
audit=json.loads((p.ROOT/'prepared/audit.json').read_text())
norm=json.loads((p.ROOT/'prepared/normalization.json').read_text())
mom={n:[0,0.,0.] for n in p.CORE+p.AUX}; counts={}; patients=[]; disturbances=0
for split in ['train','validation']:
 ds=p.LoopDataset(split=split,mode='retrospective_multimodal')
 counts[split]={'patients':int(ds.index.patient_id.nunique()),'samples':len(ds)}
 for pid,sub in ds.index.groupby('patient_id'):
  f,raw,allowed=ds._patient(pid); keep=p.history_union(len(f),sub.row_index.to_numpy())
  if split=='train':
   for j,n in enumerate(p.CORE+p.AUX):
    v=raw[0][keep & raw[1][:,j],j]; mom[n][0]+=len(v);mom[n][1]+=float(v.sum());mom[n][2]+=float(v@v)
  row=int(sub.row_index.iloc[len(sub)//2]); state=p.encode_history(raw,row,norm,'retrospective_multimodal')
  assert state.shape==(72,22) and np.isfinite(state).all()
  # Mutate only future event values and prove history unchanged.
  ff=f.copy(); ff.loc[ff.index[row+1:],p.CORE+p.AUX]=999
  raw2=p.raw_features(ff,'retrospective_multimodal')
  assert np.array_equal(state,p.encode_history(raw2,row,norm,'retrospective_multimodal')); disturbances+=1
  patients.append({'patient':pid,'split':split,'samples':len(sub),'food_observed_history_rows':int((keep&raw[1][:,3]).sum()),'exercise_observed_history_rows':int((keep&raw[1][:,4]).sum())})
  if len(patients)%25==0: print('audited',len(patients),flush=True)
for n,(count,total,sq) in mom.items():
 mean=total/count; sd=max(sq/count-mean*mean,0)**.5; sd=sd if sd>1e-12 else 1
 assert count==norm[n]['observed_training_rows']
 assert np.isclose(mean,norm[n]['mean'],rtol=1e-10) and np.isclose(sd,norm[n]['scale'],rtol=1e-9)
assert counts['train']=={'patients':225,'samples':1653421}
old=p.ROOT.parent/'训练管线_v1/prepared'
unchanged=all(p.sha(p.ROOT/'prepared'/rel)==p.sha(old/rel) for rel in audit['artifact_hashes']) if old.exists() else None
result={'counts':counts,'state_shape':[72,22],'normalization_recomputed_train_only':True,'normalization':norm,'patient_future_perturbations_passed':disturbances,'source_hash_verified':len(patients),'indices_and_statistics_identical_to_v1':unchanged,'feature_presence_filter':False,'clinical_ready':False,'contract_sha256':p.sha(p.ROOT/'contract.json'),'patients':patients}
(out/'multimodal_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='patients'},indent=2))
