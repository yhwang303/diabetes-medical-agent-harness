"""Patient-macro paired control tables. Refuse unmatched evaluation contracts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
KEYS=['tir_lower_bound_pct','tir_upper_bound_pct','tir_observed_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','cv_pct','lbgi','hbgi','coverage_pct','hypo_events_per_observed_day']
CGM_KEYS=['tir_observed_pct','tbr70_pct','tbr54_pct','coverage_pct']

def write_csv(path,rows):
 if not rows:return
 with path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def paired_interval(pairs,seed=91626):
 """Resample patient clusters and paired scenarios within each sampled cluster."""
 rng=np.random.default_rng(seed);patients=sorted(pairs);values=[]
 for _ in range(5000):
  selected=rng.choice(patients,len(patients),replace=True);means=[]
  for p in selected:
   data=np.array(pairs[p]);means.append(float(rng.choice(data,len(data),replace=True).mean()))
  values.append(float(np.mean(means)))
 return np.quantile(values,[.025,.975]).tolist()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--name',required=True);args=ap.parse_args();assert args.reference in args.runs
 runs={};configs=[]
 for name in args.runs:
  folder=ROOT/'results'/name;summary=json.loads((folder/'summary.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());assert summary['status']=='completed'
  configs.append(manifest['config']);runs[name]={(r['patient'],r['group'],r['seed']):r for r in summary['episodes']}
 assert all(c==configs[0] for c in configs),'Different scenario/time/data contracts cannot share a comparison table'
 assert all(set(r)==set(runs[args.reference]) for r in runs.values()),'Missing or unmatched episodes'
 rows=[];patient_rows=[];comparisons=[]
 for name,episodes in runs.items():
  for group in sorted(set(k[1] for k in episodes)):
   selected=[r for key,r in episodes.items() if key[1]==group];patients=sorted(set(r['patient'] for r in selected));by_patient=[]
   for p in patients:
    records=[r for r in selected if r['patient']==p];values={'run':name,'group':group,'patient':p,'episodes':len(records),'failure_pct':100*float(np.mean([r['metrics']['failed'] for r in records]))}
    for key in KEYS:
     available=[r['metrics']['bg'].get(key) for r in records];available=[x for x in available if x is not None];values[key]=float(np.mean(available)) if available else None
    for key in ['basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day']:
     available=[r['metrics'][key] for r in records if r['metrics'][key] is not None];values[key]=float(np.mean(available)) if available else None
    for key in CGM_KEYS:
     available=[r['metrics']['cgm'].get(key) for r in records];available=[x for x in available if x is not None];values['cgm_'+key]=float(np.mean(available)) if available else None
    by_patient.append(values);patient_rows.append(values)
   row={'run':name,'group':group,'patients':len(patients),'episodes':len(selected),'failures':sum(r['metrics']['failed'] for r in selected)}
   for key in ['failure_pct']+KEYS+['basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day']+['cgm_'+k for k in CGM_KEYS]:
    a=np.array([r[key] for r in by_patient if r[key] is not None]);row[key+'_mean']=float(a.mean()) if len(a) else None;row[key+'_patient_sd']=float(a.std(ddof=1)) if len(a)>1 else None;row[key+'_patients_with_observations']=len(a)
   rows.append(row)
   if name!=args.reference:
    comparison={'run':name,'reference':args.reference,'group':group,'patients':len(patients),'interval_excludes_training_seed_variation':True}
    for metric in ['tir_lower_bound_pct','tbr70_pct','tbr54_pct','failure_pct']:
     pairs={p:[] for p in patients}
     for key,r in episodes.items():
      if key[1]!=group:continue
      ref=runs[args.reference][key]
      if metric=='failure_pct':difference=100*(int(r['metrics']['failed'])-int(ref['metrics']['failed']))
      else:
       a=r['metrics']['bg'].get(metric);b=ref['metrics']['bg'].get(metric)
       if a is None or b is None:continue
       difference=a-b
      pairs[key[0]].append(difference)
     pairs={p:v for p,v in pairs.items() if v}
     comparison[metric]={'patient_macro_delta':float(np.mean([np.mean(v) for v in pairs.values()])),'paired_cluster_bootstrap95':paired_interval(pairs),'paired_patients':len(pairs)} if pairs else None
    bounds={p:[] for p in patients}
    for key,r in episodes.items():
     if key[1]!=group:continue
     a=r['metrics']['bg'];b=runs[args.reference][key]['metrics']['bg'];bounds[key[0]].append([a['tir_lower_bound_pct']-b['tir_upper_bound_pct'],a['tir_upper_bound_pct']-b['tir_lower_bound_pct']])
    comparison['paired_delta_tir_missing_outcome_bounds']=np.mean([np.mean(v,axis=0) for v in bounds.values()],axis=0).tolist()
    comparisons.append(comparison)
 out=ROOT/'paper_tables'/args.name;out.mkdir(parents=True,exist_ok=True);write_csv(out/'main.csv',rows);write_csv(out/'patients.csv',patient_rows);(out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False));(out/'contract.json').write_text(json.dumps({'config':configs[0],'reference':args.reference,'runs':args.runs,'primary_effect':'paired patient-macro standard TIR difference when complete; paired missing-outcome bounds otherwise. Lower-bound contrast is a separate failure-aware diagnostic, not the bound on the true difference.','required_companions':'upper bound, coverage, TBR54/70 and failure; observed-only risk is conditional on trajectory availability','bootstrap':'5000 paired two-level resamples, patient cluster then scenario, seed91626','training_seed':260915},indent=2));print(json.dumps({'groups':len(rows),'patient_rows':len(patient_rows),'comparisons':len(comparisons)}))

if __name__=='__main__':main()
