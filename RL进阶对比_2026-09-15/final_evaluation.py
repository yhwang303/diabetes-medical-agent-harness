"""Single frozen sealed evaluation. Does not weaken the dev-only entrypoint."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,datetime,concurrent.futures,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCES=['sim_eval.py','observable_history.py','numpy_policy.py','batch_sim.py','final_evaluation.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(rows):
 out={}
 for m in sorted({r['method'] for r in rows}):
  rs=[r for r in rows if r['method']==m];pids=sorted({r['patient'] for r in rs})
  out[m]={k:float(np.mean([np.mean([r[k] for r in rs if r['patient']==p]) for p in pids])) for k in ['tir_pct','tbr70_pct','tbr54_pct','mean_risk']}
  out[m]['failures']=sum(r['failed'] for r in rs)
 return out

def checks(rows,contract):
 s=summary(rows);a=s['Ours_bound040'];rules=contract['research_acceptance'];out={}
 best=max(['ReBRAC_BC100','FQL_alpha100'],key=lambda m:s[m]['tir_pct'])
 for ref,key in [('BC','delta_TIR_vs_BC_min_pp'),('nominal','delta_TIR_vs_nominal_min_pp'),(best,'delta_TIR_vs_best_recent_RL_min_pp')]:
  v=a['tir_pct']-s[ref]['tir_pct'];out[key]={'observed':v,'threshold':rules[key],'passed':v>=rules[key]}
 for metric,key in [('tbr70_pct','TBR70_increase_vs_nominal_and_BC_max_pp'),('tbr54_pct','TBR54_increase_vs_nominal_and_BC_max_pp')]:
  v=max(a[metric]-s[r][metric] for r in ['nominal','BC']);out[key]={'observed':v,'threshold':rules[key],'passed':v<=rules[key]}
 out['terminal_failures_allowed']={'observed':a['failures'],'threshold':rules['terminal_failures_allowed'],'passed':a['failures']==0}
 pids=sorted({r['patient'] for r in rows});d=np.array([np.mean([r['tir_pct'] for r in rows if r['patient']==p and r['method']=='Ours_bound040'])-np.mean([r['tir_pct'] for r in rows if r['patient']==p and r['method']=='BC']) for p in pids])
 ci=np.quantile(np.random.default_rng(9981).choice(d,size=(10000,len(d)),replace=True).mean(1),[.025,.975]).tolist();key='paired_patient_bootstrap95_delta_TIR_vs_BC_lower_min_pp'
 out[key]={'observed':ci[0],'ci95_pp':ci,'patient_differences_pp':d.tolist(),'threshold':rules[key],'passed':ci[0]>=rules[key]}
 return {'metrics':s,'checks':out,'all_passed':all(x['passed'] for x in out.values()),'training_seed_count':1,'random_initialization_robustness_proven':False,'clinical_ready':False}

def freeze():
 contract=json.loads((ROOT/'simulation_contract_v3_single_seed.json').read_text());d=ROOT/'results/D23_full_development_comparison';completion=json.loads((d/'completion.json').read_text());rows=json.loads((d/'summary.json').read_text());assert completion['jobs']==len(rows)==252
 spec=json.loads((ROOT/'configs/D23_full_development_comparison.json').read_text());expected={(m,p,s,b) for m in spec['policies'] for p in contract['dev_patients'] for s in contract['dev_seeds'] for b in contract['bolus_factors']};assert {(r['method'],r['patient'],r['seed'],r['factor']) for r in rows}==expected
 assessed=checks(rows,contract);assert assessed['all_passed'];(ROOT/'results/full_development_assessment.json').write_text(json.dumps(assessed,indent=2))
 # Check every reused pilot trajectory; repeats are reproducibility checks, never extra independent samples.
 previous={'nominal':('D01_pilot50k_dev','nominal'),'BC':('D01_pilot50k_dev','BC'),'ReBRAC_BC100':('D04_rebrac_bc100_dev','ReBRAC_BC100'),'Ours_bound040':('D21_bounded040_dev','Ours_bound040')}
 # Stage/method names are resolved from original provenance for the O07 candidate.
 for p in sorted((ROOT/'results').glob('D21*')):
  prov=json.loads((p/'provenance.json').read_text());previous['Ours_bound040']=(p.name,next(iter(prov['weights'])))
 repeat=[]
 for method,(stage,oldmethod) in previous.items():
  old=json.loads((ROOT/'results'/stage/'summary.json').read_text())
  for r in rows:
   if r['method']!=method or r['seed']!=3101:continue
   match=next(x for x in old if x['method']==oldmethod and all(x[k]==r[k] for k in ['patient','seed','factor']))
   a=json.loads((ROOT/r['file']).read_text());b=json.loads((ROOT/match['file']).read_text());assert a['records']==b['records'];repeat.append(r['file'])
 weights={}
 for method,path in spec['policies'].items():
  if path:
   meta=json.loads((ROOT/path).with_suffix('.json').read_text());assert meta['step']==50000 and meta['config']['seed']==260915 and sha(ROOT/path)==meta['export_sha256'];weights[method]={'path':path,'sha256':sha(ROOT/path),'metadata_sha256':sha((ROOT/path).with_suffix('.json'))}
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'name':'T01_frozen_sealed_comparison','contract':'simulation_contract_v3_single_seed.json','contract_sha256':sha(ROOT/'simulation_contract_v3_single_seed.json'),'candidate':'Ours_bound040','policies':spec['policies'],'weights':weights,'source_sha256':{s:sha(ROOT/s) for s in SOURCES},'patients':contract['sealed_patients'],'seeds':contract['sealed_seeds'],'bolus_factors':contract['bolus_factors'],'training_seed':260915,'training_steps_all':50000,'selection_evidence_sha256':sha(d/'summary.json'),'development_assessment_sha256':sha(ROOT/'results/full_development_assessment.json'),'repeated_pilot_trajectories_exact':len(repeat),'test_tuning_allowed':False,'clinical_ready':False}
 path=ROOT/'configs/T01_frozen_sealed_comparison.json'
 with path.open('x') as f:json.dump(plan,f,indent=2)
 print(json.dumps({'frozen':str(path),'sha256':sha(path),'development':assessed},indent=2))

def evaluate():
 from batch_sim import evaluate_job
 path=ROOT/'configs/T01_frozen_sealed_comparison.json';plan=json.loads(path.read_text());cpath=ROOT/plan['contract'];assert sha(cpath)==plan['contract_sha256'];contract=json.loads(cpath.read_text())
 assert plan['patients']==contract['sealed_patients'] and plan['seeds']==contract['sealed_seeds'] and plan['bolus_factors']==contract['bolus_factors'];assert not set(plan['patients'])&set(contract['dev_patients'])
 for s,v in plan['source_sha256'].items():assert sha(ROOT/s)==v,s
 for m,w in plan['weights'].items():assert sha(ROOT/w['path'])==w['sha256'] and sha((ROOT/w['path']).with_suffix('.json'))==w['metadata_sha256'],m
 dest=ROOT/'results'/plan['name'];dest.mkdir(exist_ok=False);jobs=[]
 for m,path in plan['policies'].items():
  for p in plan['patients']:
   for s in plan['seeds']:
    for b in plan['bolus_factors']:jobs.append((m,p,s,b,str(ROOT/path) if path else None,str(dest/('%s_p%d_s%d_b%.1f.json'%(m,p,s,b)))))
 (dest/'provenance.json').write_text(json.dumps({'frozen_plan':plan,'frozen_plan_sha256':sha(ROOT/'configs/T01_frozen_sealed_comparison.json'),'jobs':len(jobs),'sealed_once':True},indent=2));begin=time.time();rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
  for f in concurrent.futures.as_completed([pool.submit(evaluate_job,j) for j in jobs]):
   r=f.result();rows.append(r);(dest/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'completed':len(rows),'total':len(jobs),'method':r['method'],'patient':r['patient'],'seed':r['seed'],'factor':r['factor']}),flush=True)
 (dest/'completion.json').write_text(json.dumps({'jobs':len(rows),'wall_seconds':time.time()-begin,'clinical_ready':False},indent=2));assessment=checks(rows,contract);(dest/'assessment.json').write_text(json.dumps(assessment,indent=2));print(json.dumps(assessment,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='evaluate':evaluate()
 else:raise ValueError('freeze or evaluate required')
