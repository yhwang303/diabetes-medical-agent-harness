"""Episode-isolated observed histories and action-conditioned future labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(P/'RL进阶对比_2026-09-15'))
from observable_history import History


def windows(raw,horizon=72,stride=3):
    rows=raw['records']
    history=History();histories=[]
    for row in rows:
        history.append(row['minute'],row['cgm_mg_dl'],row['delivered_basal_u_h']/12,
                       row['bolus_u'] if row['bolus_u']>0 else None,
                       row['meal_g'] if row['meal_g']>0 else None)
        histories.append(history.state() if len(history.rows)>=72 else None)
    anchor=rows[71]['delivered_basal_u_h']
    fields={k:[] for k in ['history','anchor_u_h','actions_u_h','target_cgm_mg_dl',
                          'target_bg_mg_dl','mask','origin_minute']}
    for i in range(71,len(rows)-1,stride):
        future=rows[i+1:i+1+horizon]
        mask=np.arange(horizon)<len(future)
        actions=np.full(horizon,anchor,dtype=np.float32)
        cgm=np.zeros(horizon,dtype=np.float32);bg=np.zeros(horizon,dtype=np.float32)
        for j,row in enumerate(future):
            actions[j]=row['delivered_basal_u_h']
            cgm[j]=row['cgm_mg_dl'];bg[j]=row['bg_mg_dl']
        assert np.isfinite(cgm[mask]).all() and np.isfinite(bg[mask]).all()
        fields['history'].append(histories[i]);fields['anchor_u_h'].append(anchor)
        fields['actions_u_h'].append(actions);fields['target_cgm_mg_dl'].append(cgm)
        fields['target_bg_mg_dl'].append(bg);fields['mask'].append(mask)
        fields['origin_minute'].append(rows[i]['minute'])
    return {k:np.asarray(v,dtype=bool if k=='mask' else np.float32) for k,v in fields.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True)
    ap.add_argument('--name',required=True);ap.add_argument('--stride',type=int,default=3)
    args=ap.parse_args();source=R/'results'/args.source
    summary=json.loads((source/'summary.json').read_text());assert summary['status']=='completed'
    provenance=json.loads((source/'manifest.json').read_text())
    split=provenance['config']['split'];assert split in ('train','world_validation')
    out=R/'cache'/args.name;out.mkdir(parents=True,exist_ok=False)
    counts=[]
    for episode in summary['episodes']:
        path=source/(episode['key']+'.json');raw=json.loads(path.read_text())
        if raw['failure_reason'] not in (None,'native_environment_done'):
            raise ValueError('Technical failure is not a valid training episode')
        packed=windows(raw,stride=args.stride)
        dest=out/(episode['key']+'.npz');np.savez_compressed(dest,**packed)
        counts.append(dict(file=dest.name,origins=len(packed['history']),
                           full_windows=int(packed['mask'].all(1).sum()),
                           future_low70_origins=int(((packed['target_bg_mg_dl']<70)&packed['mask']).any(1).sum()),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),job=raw['job']))
    manifest=dict(status='completed',split=split,source=args.source,stride=args.stride,horizon=72,
                  units='mg/dL and U/h',patient_identity_used_as_feature=False,
                  source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  future_meals_or_bolus_in_model_input=False,episodes=counts,
                  origins=sum(x['origins'] for x in counts))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status='completed',split=split,origins=manifest['origins'],episodes=len(counts))),flush=True)


if __name__=='__main__':main()
