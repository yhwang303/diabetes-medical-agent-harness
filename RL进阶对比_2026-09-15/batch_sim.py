"""Paired development evaluation; frozen patient/scenario assignments and checkpoint hashes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import argparse,json,hashlib,concurrent.futures,time,traceback
from pathlib import Path
from sim_eval import simulate,ROOT

def evaluate_job(job):
 method,pid,seed,factor,policy,out=job;path=Path(out)
 try:
  result=simulate(pid,seed,policy,3,factor,'learned' if policy else 'nominal');result['method']=method
  path.write_text(json.dumps(result,allow_nan=False));return {'method':method,'patient':pid,'seed':seed,'factor':factor,**result['metrics'],'file':str(path.relative_to(ROOT))}
 except Exception as e:
  path.with_suffix('.failure.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()}));raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();spec=json.loads(Path(args.spec).read_text());dest=ROOT/'results'/spec['name'];dest.mkdir(parents=True,exist_ok=True)
 if (dest/'provenance.json').exists():raise RuntimeError('Evaluation name already exists; preserve artifacts')
 contract=json.loads((ROOT/'simulation_contract_v1.json').read_text());jobs=[];weights={}
 extension_info=None
 if spec.get('protocol_extension'):
  ep=ROOT/spec['protocol_extension'];extension=json.loads(ep.read_text());parent_sha=hashlib.sha256((ROOT/'simulation_contract_v1.json').read_bytes()).hexdigest()
  assert extension['parent_contract_sha256']==parent_sha
  for key in ['dev_patients','dev_seeds','sealed_patients','sealed_seeds','bolus_factors','normalization_sha256']:assert extension[key]==contract[key],key
  allowed_seed_changes={'training_seeds_min','seeds_with_TIR_improvement_vs_BC_min'} if extension['version']=='sim-eval-v3-single-training-seed' else set()
  for key,value in contract['research_acceptance'].items():
   if key not in allowed_seed_changes:assert extension['research_acceptance'][key]==value,key
  if allowed_seed_changes:assert all(extension['research_acceptance'][key]==1 for key in allowed_seed_changes)
  extension_info={'path':spec['protocol_extension'],'sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),'parent_sha256':parent_sha,'unchanged_performance_thresholds_verified':True,'user_authorized_training_seed_count':extension.get('training_seed_scope')}

 for method,relative in spec['policies'].items():
  policy=str(ROOT/relative) if relative else None
  if policy:
   p=Path(policy);meta=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['export_sha256'];assert meta['numpy_torch_max_abs_action_normalized_error']<2e-4;weights[method]=meta
  for pid in spec['patients']:
   assert pid in contract['dev_patients']
   for seed in spec['seeds']:
    assert seed in contract['dev_seeds']
    for factor in contract['bolus_factors']:jobs.append((method,pid,seed,factor,policy,str(dest/('%s_p%d_s%d_b%.1f.json'%(method,pid,seed,factor)))))
 provenance={'spec':spec,'protocol_extension':extension_info,'contract_sha256':hashlib.sha256((ROOT/'simulation_contract_v1.json').read_bytes()).hexdigest(),'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['sim_eval.py','observable_history.py','numpy_policy.py','batch_sim.py']},'weights':weights,'jobs':len(jobs),'clinical_ready':False};(dest/'provenance.json').write_text(json.dumps(provenance,indent=2));begin=time.time();rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(evaluate_job,j) for j in jobs]
  for f in concurrent.futures.as_completed(futures):
   row=f.result();rows.append(row);(dest/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'completed':len(rows),'total':len(jobs),**row}),flush=True)
 (dest/'completion.json').write_text(json.dumps({'jobs':len(rows),'wall_seconds':time.time()-begin,'clinical_ready':False},indent=2))
if __name__=='__main__':main()
