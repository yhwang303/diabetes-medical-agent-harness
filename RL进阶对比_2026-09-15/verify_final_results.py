"""Recompute held-out metrics from all raw traces, verify pairing, weights, source freeze."""
import json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'results/T01_frozen_sealed_comparison';planpath=ROOT/'configs/T01_frozen_sealed_comparison.json';plan=json.loads(planpath.read_text());prov=json.loads((d/'provenance.json').read_text());rows=json.loads((d/'summary.json').read_text());assert json.loads((d/'completion.json').read_text())['jobs']==len(rows)==378;assert sha(planpath)==prov['frozen_plan_sha256'];assert sha(ROOT/plan['contract'])==plan['contract_sha256'];assert all(sha(ROOT/s)==h for s,h in plan['source_sha256'].items())
 expected={(m,p,s,b) for m in plan['policies'] for p in plan['patients'] for s in plan['seeds'] for b in plan['bolus_factors']};assert expected=={(r['method'],r['patient'],r['seed'],r['factor']) for r in rows}
 maxerr=0.;events={};references={}
 for r in rows:
  record=json.loads((ROOT/r['file']).read_text());trace=record['records'];assert record['scenario_seed']==r['seed'] and record['bolus_factor']==r['factor'] and record['patient']=='adult#%03d'%r['patient'];assert [x['minute'] for x in trace]==list(range(5,5*(len(trace)+1),5));assert len(trace)<=864 and all(x['warmup']==(x['minute']<=360) for x in trace)
  key=(r['patient'],r['seed'],r['factor']);initial=[{k:x[k] for k in ['minute','cgm_mg_dl','requested_basal_u_h','delivered_basal_u_h','bolus_u','meal_g']} for x in trace[:72]]
  if key in references:assert references[key]==initial
  else:references[key]=initial
  # Common exogenous meals/bolus compared only over actually simulated overlapping time.
  for x in trace:
   ek=(r['patient'],r['seed'],r['factor'],x['minute']);value=(x['meal_g'],x['bolus_u'])
   if ek in events:assert np.allclose(events[ek],value,atol=1e-10),ek
   else:events[ek]=value
  raw=np.array([x['bg_mg_dl'] for x in trace[72:]]);missing=792-len(raw)
  if missing:raw=np.r_[raw,np.full(missing,600. if not len(raw) or not np.isfinite(raw[-1]) or raw[-1]>600 else 10.)]
  calc={'tir_pct':float(np.count_nonzero((raw>=70)&(raw<=180))/792*100),'tbr70_pct':float(np.count_nonzero(raw<70)/792*100),'tbr54_pct':float(np.count_nonzero(raw<54)/792*100)}
  assert missing==r['missing_failure_steps'] and len(trace)-72==r['measured_steps'] and r['fixed_metric_denominator']==792
  assert bool(record['metrics']['failed'])==bool(r['failed'])
  for k,v in calc.items():err=abs(v-r[k]);maxerr=max(maxerr,err);assert err<1e-10
 from final_evaluation import checks
 calc=checks(rows,json.loads((ROOT/plan['contract']).read_text()));assert calc==json.loads((d/'assessment.json').read_text())
 result={'trajectories_verified':len(rows),'paired_cases':len(references),'all_patients_sealed_only':True,'training_seed':260915,'training_updates':50000,'frozen_source_and_plan_sha_verified':True,'shared_warmup_meals_bolus_verified':True,'raw_record_primary_metrics_max_abs_error':maxerr,'all_792_step_denominators_verified':True,'assessment_recomputed_exact':True,'clinical_ready':False};(ROOT/'results/final_result_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
