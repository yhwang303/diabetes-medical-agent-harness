"""Offline-only extension of existing T01 traces. No baseline execution."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from control_metrics import summarize
ROOT=Path(__file__).resolve().parent

def main():
 source=ROOT.parent/'RL进阶对比_2026-09-15/results/T01_frozen_sealed_comparison';out=ROOT/'legacy_panel';out.mkdir(exist_ok=True);episodes=[];hashes={}
 for p in sorted(source.glob('*.json')):
  obj=json.loads(p.read_text())
  if 'records' not in obj:continue
  assert obj['days']==3 and obj['metrics']['fixed_metric_denominator']==792
  metrics=summarize(obj['records'],792,obj['metrics']['failed'])
  assert metrics['observed_intervals']==obj['metrics']['measured_steps']
  episodes.append({'method':obj['method'],'patient':obj['patient'],'seed':obj['scenario_seed'],'bolus_factor':obj['bolus_factor'],'old_failure_penalized_tir':obj['metrics']['tir_pct'],'metrics':metrics});hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert len(episodes)==378
 rows=[]
 for method in sorted(set(e['method'] for e in episodes)):
  selected=[e for e in episodes if e['method']==method];assert len(selected)==54
  row={'method':method,'episodes':len(selected),'patients':len(set(e['patient'] for e in selected)),'failures':sum(e['metrics']['failed'] for e in selected),'old_failure_penalized_tir_pct':float(np.mean([e['old_failure_penalized_tir'] for e in selected]))}
  for key in ['coverage_pct','tir_observed_pct','tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','cv_pct','lbgi','hbgi','hypo_events_per_observed_day','max_hypo_low_minutes']:
   values=[]
   for patient in sorted(set(e['patient'] for e in selected)):
    group=[e['metrics']['bg'].get(key) for e in selected if e['patient']==patient];group=[v for v in group if v is not None];values.append(float(np.mean(group)))
   row[key]=float(np.mean(values))
  for key in ['basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day']:
   row[key]=float(np.mean([e['metrics'][key] for e in selected if e['metrics'][key] is not None]))
  rows.append(row)
 with (out/'observed_metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (out/'episodes.json').write_text(json.dumps(episodes,allow_nan=False));(out/'summary.json').write_text(json.dumps({'scope':'exposed T01 regression panel, offline recomputation only','training_runs_added':0,'simulation_runs_added':0,'rows':rows,'warning':'Observed-only metrics in failed episodes are subject to informative early stopping; coverage, failures and TIR bounds must stay adjacent. Not a new sealed experiment.'},indent=2,allow_nan=False));(out/'provenance.json').write_text(json.dumps({'source_sha256':hashes,'metric_source_sha256':hashlib.sha256((ROOT/'control_metrics.py').read_bytes()).hexdigest()},indent=2));print(json.dumps({'episodes':len(episodes),'methods':len(rows),'new_simulation_runs':0}))

if __name__=='__main__':main()
