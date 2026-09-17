"""Stateless JSON research inference. This is not a registered Core executor or dosing publication."""
import argparse,hashlib,json,sys,datetime
from pathlib import Path
import numpy as np
from numpy_policy import NumpyPolicy

class InputRejected(ValueError):pass

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(value):
 d=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
 if d.tzinfo is None:raise InputRejected('timestamp requires timezone')
 return d

def infer(bundle,request):
 bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
 for name,expected in manifest['files_sha256'].items():
  path=(bundle/name).resolve()
  if path.parent!=bundle.resolve() or digest(path)!=expected:raise InputRejected('bundle fingerprint mismatch')
 if request['normalization_sha256']!=manifest['normalization_sha256']:raise InputRejected('normalizer mismatch')
 patient=request['patient_id'];reference=request['reference'];end=utc(request['observation_end_utc'])
 if not isinstance(patient,str) or not patient or reference['patient_id']!=patient:raise InputRejected('reference patient mismatch')
 start=utc(reference['history_start_utc']);last=utc(reference['history_end_utc'])
 if last>end or (last-start).total_seconds()!=71*300:raise InputRejected('reference history timing mismatch')
 if (end-last).total_seconds()>manifest['reference_max_age_hours']*3600:raise InputRejected('reference outside evaluated age scope')
 basal=np.asarray(reference['previous_5min_basal_u'],dtype='float64')
 if basal.shape!=(72,) or not np.isfinite(basal).all() or np.any(basal<0):raise InputRejected('reference requires72 finite observed nonnegative intervals')
 anchor=float(basal.mean()*12)
 if not 0<=anchor<=20:raise InputRejected('reference outside model action range')
 history=np.asarray(request['normalized_history_72x22'],dtype='float32')
 if history.shape!=(72,22) or not np.isfinite(history).all():raise InputRejected('history shape/nonfinite')
 if not np.isin(history[:,5:10],[0,1]).all() or not np.isin(history[:,15:20],[0,1]).all():raise InputRejected('history masks must be binary')
 if not np.allclose(history[:,20],1/12,atol=1e-6) or not np.allclose(history[:,21],np.arange(-71,1)/12,atol=1e-6):raise InputRejected('history time encoding mismatch')
 if history[-1,5]!=1:raise InputRejected('current CGM must be observed')
 policy=NumpyPolicy(bundle/'policy.npz',bundle/'normalization.json')
 if policy.cfg['algorithm']!='rebrac' or policy.cfg['seed']!=260915:raise InputRejected('unexpected research model')
 state=np.concatenate([history.reshape(1,-1),np.array([[anchor/10-1]],dtype='float32')],-1)
 action=float(10*(policy(state)[0,0]+1));limit=float(policy.cfg['residual_limit_u_h'])
 if not np.isfinite(action) or not 0<=action<=20 or abs(action-anchor)>limit+1e-5:raise InputRejected('invalid action')
 return {'schema':'research-rl-result-v1','patient_id':patient,'observation_end_utc':request['observation_end_utc'],'request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'model_id':manifest['model_id'],'model_sha256':manifest['files_sha256']['policy.npz'],'normalization_sha256':manifest['normalization_sha256'],'reference_u_h':anchor,'reference_history_end_utc':reference['history_end_utc'],'basal_rate_u_h':action,'action_duration_minutes':5,'basal_interval_u':action*5/60,'model_output_quantized':False,'clinical_ready':False,'registered_with_harness_core':False,'scope':'Research candidate inference only; requires Core source/applicability/evidence checks before any future integration.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);args=ap.parse_args()
 try:
  raw=sys.stdin.buffer.read(131073)
  if len(raw)>131072:raise InputRejected('input too large')
  request=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(InputRejected('nonfinite JSON')))
  result=infer(args.bundle,request);print(json.dumps({'ok':True,'result':result},allow_nan=False))
 except (InputRejected,KeyError,ValueError,TypeError,OSError) as e:
  print(json.dumps({'ok':False,'error':str(e),'clinical_ready':False}));raise SystemExit(2)
if __name__=='__main__':main()
