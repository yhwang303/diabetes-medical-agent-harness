from pathlib import Path
import sys,json
import numpy as np
from data import Data,ROOT
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline as p
checked=0;future=0
for split in ['train','validation']:
 data=Data(split);ds=p.LoopDataset(split=split,mode='retrospective_multimodal');assert data.total==len(ds)
 groups=ds.index.groupby('patient_id').indices
 for pid,d in data.patients.items():
  positions=np.unique(np.linspace(0,len(d['starts'])-1,min(3,len(d['starts']))).astype(int));bb=data.batch(pid,positions,12,np.random.default_rng(73))
  f,raw,allowed=ds._patient(pid)
  for j,pos in enumerate(positions):
   item=ds[int(groups[pid][pos])];assert int(bb['row'][j])==item['row_index']
   np.testing.assert_allclose(bb['state'][j],item['state'],atol=1e-6,rtol=1e-6)
   np.testing.assert_allclose(bb['next_state'][j],p.encode_history(raw,int(bb['row'][j]+bb['k'][j]+1),ds.normalizer,'retrospective_multimodal'),atol=1e-6,rtol=1e-6)
   n=int(bb['length'][j]);s=int(bb['row'][j]);np.testing.assert_array_equal(bb['mask'][j],np.arange(12)<n)
   np.testing.assert_allclose(bb['target'][j,:n],f.cgm_mmol_l.iloc[s+1:s+n+1].to_numpy(),rtol=1e-6)
   np.testing.assert_allclose(bb['action'][j,:n],f.basal_action_u_h.iloc[s:s+n].to_numpy(),rtol=1e-6)
   assert allowed[s:s+n].all() and (f.episode_id.iloc[s:s+n]==f.episode_id.iloc[s]).all()
   checked+=1;future+=n
 result={'samples_compared':checked,'valid_future_steps_checked':future,'packed_matches_authoritative_pipeline':True,'tolerance_float32':1e-6,'clinical_ready':False}
 (ROOT/'results/packed_verification.json').write_text(json.dumps(result,indent=2))
 print(split,checked,flush=True)
