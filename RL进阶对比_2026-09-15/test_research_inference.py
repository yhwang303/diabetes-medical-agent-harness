"""Real evaluated trajectory -> fresh JSON subprocess, with rejection cases."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,subprocess,copy,tempfile,shutil,hashlib,datetime,time
from pathlib import Path
import numpy as np
from observable_history import History
ROOT=Path(__file__).resolve().parent

def main():
 bundle=ROOT/'artifacts/O07_research_candidate';manifest=json.loads((bundle/'manifest.json').read_text());trace=json.loads((ROOT/'results/D23_full_development_comparison/Ours_bound040_p1_s3101_b1.0.json').read_text());records=trace['records'];history=History();base=datetime.datetime(2020,1,1,tzinfo=datetime.timezone.utc)
 def stamp(m):return (base+datetime.timedelta(minutes=m)).isoformat()
 ref={'patient_id':trace['patient'],'history_start_utc':stamp(5),'history_end_utc':stamp(360),'previous_5min_basal_u':[r['delivered_basal_u_h']/12 for r in records[:72]]}
 requests=[]
 for index,r in enumerate(records):
  history.append(r['minute'],r['cgm_mg_dl'],r['delivered_basal_u_h']/12,r['bolus_u'] if r['bolus_u']>0 else None,r['meal_g'] if r['meal_g']>0 else None)
  if index in [71,287,575,862]:
   request={'patient_id':trace['patient'],'normalization_sha256':manifest['normalization_sha256'],'observation_end_utc':stamp(r['minute']),'reference':ref,'normalized_history_72x22':history.state().tolist()};requests.append((request,records[index+1]['requested_basal_u_h']))
 def call(req,b=bundle):
  start=time.perf_counter();p=subprocess.run([sys.executable,str(b/'research_inference.py'),'--bundle',str(b)],input=json.dumps(req).encode(),capture_output=True,timeout=30);return p,json.loads(p.stdout),time.perf_counter()-start
 verified=[]
 for req,expected in requests:
  p,out,wall=call(req);assert p.returncode==0,out;actual=out['result']['basal_rate_u_h'];assert abs(actual-expected)<2e-5,(actual,expected);assert abs(out['result']['basal_interval_u']-actual/12)<1e-12;verified.append({'observation_end':req['observation_end_utc'],'evaluated_action_u_h':expected,'fresh_process_action_u_h':actual,'absolute_error_u_h':abs(actual-expected),'wall_seconds':wall})
 request=requests[0][0];(bundle/'example_request.json').write_text(json.dumps(request));(bundle/'example_result.json').write_text(json.dumps(call(request)[1],indent=2));negative=[]
 for case in ['patient','normalizer','future_reference','stale_reference','shape','mask','nonfinite','time_encoding']:
  bad=copy.deepcopy(request)
  if case=='patient':bad['reference']['patient_id']='adult#002'
  elif case=='normalizer':bad['normalization_sha256']='wrong'
  elif case=='future_reference':bad['observation_end_utc']=stamp(355)
  elif case=='stale_reference':bad['observation_end_utc']=stamp(4400)
  elif case=='shape':bad['normalized_history_72x22'].pop()
  elif case=='mask':bad['normalized_history_72x22'][0][5]=.5
  elif case=='nonfinite':bad['normalized_history_72x22'][0][0]=float('nan')
  elif case=='time_encoding':bad['normalized_history_72x22'][0][20]=5
  p,out,wall=call(bad);assert p.returncode==2 and out['ok'] is False,(case,out);negative.append({'case':case,'error':out['error']})
 with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
  altered=Path(tmp)/'copy';shutil.copytree(bundle,altered);p=altered/'policy.npz'
  with p.open('ab') as f:f.write(b'tamper')
  p,out,_=call(request,altered);assert p.returncode==2 and out['ok'] is False;negative.append({'case':'weight_tamper_copy','error':out['error']})
 result={'actual_development_trajectory_sha256':hashlib.sha256((ROOT/'results/D23_full_development_comparison/Ours_bound040_p1_s3101_b1.0.json').read_bytes()).hexdigest(),'real_trajectory_fresh_subprocess_checks':verified,'negative_cases':negative,'manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),'python':sys.version,'numpy':np.__version__,'core_registration':False,'clinical_ready':False}
 (ROOT/'results/research_inference_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
