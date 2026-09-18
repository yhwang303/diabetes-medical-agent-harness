"""Open held-out Loop only after a final model manifest has been frozen."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Loop数据集/训练管线_v2'))
import pipeline

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text())
    assert freeze['no_further_selection'] is True
    ds=pipeline.LoopDataset(split='sealed_test',mode='retrospective_multimodal',allow_sealed_test=True)
    out=ROOT/'final_forecast_data';out.mkdir(exist_ok=False)
    records=[]
    training={pid for pid,r in ds.audit['source_files'].items() if r['split'] in ('train','validation')}
    assert not training&set(ds.index.patient_id)
    for pid,index in ds.index.groupby('patient_id'):
        frame,raw,_=ds._patient(pid);values,observed,age,known=raw
        mean=np.array([ds.normalizer[k]['mean'] for k in pipeline.CORE+pipeline.AUX])
        scale=np.array([ds.normalizer[k]['scale'] for k in pipeline.CORE+pipeline.AUX])
        features=np.concatenate([np.where(observed,(values-mean)/scale,0),observed,age,known],1).astype('float32')
        study=frame.in_study_window.to_numpy(bool)
        label_observed=observed[:,0]&(study|np.r_[False,study[:-1]])
        starts=index.row_index.to_numpy(dtype='int32');segments=frame.cgm_segment_id.to_numpy(dtype='int32')
        assert label_observed[starts+1].all() and (segments[starts]==segments[starts+1]).all()
        file=out/(pid+'.npz')
        np.savez_compressed(file,features=features,glucose=frame.cgm_mmol_l.to_numpy('float32'),starts=starts,episode=segments,observed=label_observed)
        records.append({'patient_id':pid,'samples':len(starts),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    result={'split':'sealed_test','patients':records,'sample_count':sum(r['samples'] for r in records),
            'patient_count':len(records),'patient_disjoint_verified':True,'all_frozen_origins_preserved':True,
            'normalizer_sha256':pipeline.sha(pipeline.ROOT/'prepared/normalization.json'),
            'final_freeze_sha256':hashlib.sha256((ROOT/'configs/final_freeze.json').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='patients'}))

if __name__=='__main__':main()
