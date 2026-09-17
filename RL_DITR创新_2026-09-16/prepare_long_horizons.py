"""Extend only valid supervision lengths to 4h; preserve every existing origin."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline as pipeline

def main():
 for split in ['train','validation']:
  ds=pipeline.LoopDataset(split=split,mode='retrospective_multimodal');out=ROOT/'long_horizons'/split;out.mkdir(parents=True,exist_ok=True);records=[]
  for pid,index in ds.index.groupby('patient_id'):
   frame,raw,allowed=ds._patient(pid);starts=index.row_index.to_numpy();episodes=frame.episode_id.to_numpy();n=len(frame);length=np.zeros(len(starts),np.int16);valid=np.ones(len(starts),bool)
   for h in range(48):
    row=np.minimum(starts+h,n-1);valid&=(starts+h<n-1)&allowed[row]&(episodes[row]==episodes[starts]);length+=valid
   old_path=ROOT.parent/'RL训练_2026-09-15/packed'/split/(pid+'.npz')
   with np.load(old_path) as old:
    np.testing.assert_array_equal(starts,old['starts']);np.testing.assert_array_equal(np.minimum(length,12),old['length'])
   path=out/(pid+'.npy');np.save(path,length);records.append({'patient':pid,'origins':len(starts),'horizon_counts':{str(h*5):int((length>=h).sum()) for h in [1,3,6,12,24,48]},'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  result={'split':split,'all_origins_preserved':True,'original_12step_lengths_exact':True,'patients':records,'total_origins':sum(r['origins'] for r in records),'scope':'additional valid future-length index only; no new eligibility filter, no sealed test read'}
  (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'split':split,'patients':len(records),'origins':result['total_origins'],'horizon_counts':{str(h*5):sum(r['horizon_counts'][str(h*5)] for r in records) for h in [1,3,6,12,24,48]}}),flush=True)

if __name__=='__main__':main()
