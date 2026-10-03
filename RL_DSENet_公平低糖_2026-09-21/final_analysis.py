"""Recompute raw outcomes, paired patient statistics and auditable table payload."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/'RL_DITR创新_2026-09-16'))
from control_metrics import summarize
from aggregate import aggregate,joint
G=['tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','mean_mg_dl','sd_mg_dl','cv_pct','lbgi','hbgi','coverage_pct']
D=['basal_u_per_observed_day','bolus_u_per_observed_day','action_tv_per_observed_day','pump_changed_fraction']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(folder,expected=None):
 a=aggregate(folder);summary=json.loads((folder/'summary.json').read_text());manifest=json.loads((folder/'manifest.json').read_text());by={};extra={};jobs=[]
 for e in summary['episodes']:
  x=json.loads((folder/(e['key']+'.json')).read_text());p=str(x['job']['patient']);by.setdefault(p,[]).append(x);jobs.append(x['job'])
  assert len([z for z in x['records'] if z['warmup']])==72
  assert x['metrics']['planned_intervals']==792
  rows=[z for z in x['records'] if not z['warmup']]
  assert all(0<=z['delivered_basal_u_h']<=20 for z in rows)
  assert len(rows)==792 or x['failure_reason'] is not None
 if expected is not None:assert sorted(a['keys'])==expected
 assert len(by)==10 and all(len(v)==6 for v in by.values())
 for k in D:
  vals=[float(np.mean([x['metrics'][k] for x in cases])) for cases in by.values()];extra[k]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))}
 extra['cgm']={k:{'mean':float(np.mean([np.mean([x['metrics']['cgm'][k] for x in cases]) for cases in by.values()]))} for k in G}
 extra['manifest_sha256']=sha(folder/'manifest.json');extra['summary_sha256']=sha(folder/'summary.json');extra['jobs']=sorted(jobs,key=lambda x:(x['patient'],x['group'],x['seed']));extra['failure_reasons']=[(e['key'],e['failure_reason']) for e in summary['episodes'] if e['failure_reason']]
 a.update(extra);return a

def paired(a,b):
 ids=sorted(a['patients']);assert ids==sorted(b['patients']);out={}
 rng=np.random.default_rng(260915);samples=rng.integers(0,len(ids),(20000,len(ids)))
 for k in G:
  delta=np.array([a['patients'][i][k]-b['patients'][i][k] for i in ids]);boot=delta[samples].mean(1)
  out[k]={'ours_minus_other':float(delta.mean()),'patient_bootstrap_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'n_patients':len(ids),'descriptive_no_multiplicity_correction':True}
 return out

def main():
 manifest=json.loads((R/'configs/final_manifest.json').read_text());assert not manifest['selection_pending'];out={'manifest':manifest,'panels':{},'paired':{}};expected=None;first_jobs=None
 for panel in manifest['panels']:
  rows={}
  for method in manifest['methods']:
   key=method['key'];folder=R/'results'/('C_'+panel+'_'+key);a=read(folder,expected)
   if expected is None:expected=a['keys'];first_jobs=a['jobs']
   assert a['jobs']==first_jobs
   em=json.loads((folder/'manifest.json').read_text());assert em['common_limit_u_h'] is None
   assert em['brake_config']==(json.loads((R/'configs/brake17.json').read_text()) if panel=='brake17' else None)
   if method['checkpoint']:assert em['checkpoint_sha256']==method['checkpoint_sha256']
   rows[key]=a
  out['panels'][panel]=rows;out['paired'][panel]={k:paired(rows['ours'],v) for k,v in rows.items() if k!='ours'}
 out['source_order_sensitivity']={}
 for key in ('fql','rebrac','lom'):
  out['source_order_sensitivity'][key]={'old':aggregate(P/'RL_DSENet_2026-09-17/revision/results'/('M_'+key)),'corrected':aggregate(R/'results'/('S_order_'+key))}
 out['new_vs_old_development']=json.loads((R/'analysis/development.json').read_text())
 out['original_weights_verified']={}
 for role,entry in json.loads((R/'configs/retained_weights.json').read_text()).items():
  path=P/'RL_DSENet_2026-09-17'/entry['path'];actual=sha(path);assert actual==entry['sha256']
  out['original_weights_verified'][role]={'path':str(path.relative_to(P)),'sha256':actual,'verified':True}
 (R/'analysis/final.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
 for panel,rows in out['panels'].items():
  with (R/'delivery'/('comparison_'+panel+'.csv')).open('w',newline='') as f:
   cols=['method','venue','year','training_information','failures','episodes','prolonged54']+[k+'_'+s for k in G+D for s in ['mean','sd']]
   w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
   for m in manifest['methods']:
    a=rows[m['key']];v={k:m[k] for k in ['venue','year','training_information']};v['method']=m['label'];v.update({k:a[k] for k in ['failures','episodes','prolonged54']});v.update({k+'_'+s:a[k][s] for k in G+D for s in ['mean','sd']});w.writerow(v)
 print(json.dumps({'status':'all_raw_scores_exact','panels':len(out['panels']),'methods':len(manifest['methods']),'episodes':sum(x['episodes'] for rows in out['panels'].values() for x in rows.values())}))
if __name__=='__main__':main()
