"""Extend frozen compact arrays for graph dynamics; retain all origins and known valid horizons."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 for split in ['train','validation']:
  source=ROOT.parent/'RL训练_2026-09-15/packed'/split;manifest=json.loads((source/'manifest.json').read_text());parts={k:[] for k in ['action_grid','glucose_grid','length']}
  for patient in manifest['patients']:
   with np.load(source/(patient['patient_id']+'.npz')) as f:
    for key,original in [('action_grid','action'),('glucose_grid','glucose'),('length','length')]:parts[key].append(f[original])
  dest=ROOT/'dynamics_data'/split;dest.mkdir(parents=True,exist_ok=True);hashes={}
  for key,arrays in parts.items():
   path=dest/(key+'.npy');np.save(path,np.concatenate(arrays));hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  assert sum(len(x) for x in parts['length'])==manifest['sample_count']
  (dest/'manifest.json').write_text(json.dumps({'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),'array_sha256':hashes,'origins':manifest['sample_count'],'horizon_steps':12,'future_events':'logged normalized bolus/carb/exercise with masks from next-row previous-interval features; retrospective only','clinical_ready':False},indent=2));print(split,manifest['sample_count'],flush=True)
if __name__=='__main__':main()
