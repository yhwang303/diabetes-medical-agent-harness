"""Untrained zero-residual control; provenance and real-policy reference-cache checks."""
import json,hashlib
from pathlib import Path
import numpy as np
from numpy_policy import NumpyPolicy
ROOT=Path(__file__).resolve().parent

def main():
 source=ROOT/'results/R31_anchor_rebrac/export_0050000.npz'
 meta=json.loads(source.with_suffix('.json').read_text())
 dest=ROOT/'results/C01_reference_only';dest.mkdir(exist_ok=True)
 target=dest/'export_0000000.npz'
 if target.exists():raise RuntimeError('Preserve existing reference control')
 with np.load(source) as f:weights={k:f[k].copy() for k in f.files}
 last=[x for x in meta['modules'] if x['type']=='Linear'][-1]['key']
 weights[last+'.weight'].fill(0);weights[last+'.bias'].fill(0)
 np.savez(target,**weights)
 meta.update(step=0,control='Fixed observed warmup reference only; no learned residual; not an RL policy',parent_export_sha256=meta['export_sha256'],export_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),training_updates=0)
 meta.pop('numpy_torch_max_abs_action_normalized_error',None)
 target.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 norm=json.loads((ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text())['basal_u_prev_5min']
 def history(rate):
  h=np.zeros((72,22),dtype='float32');h[:,1]=(rate/12-norm['mean'])/norm['scale'];h[:,6]=1;return h
 maxerr=0.
 # Trained policy's implicit six-hour reference must equal explicit reference input.
 for rate in [0.,.8,2.,4.]:
  h=history(rate);p=NumpyPolicy(source);actual=p(h);reference=p.initial_reference
  expected=NumpyPolicy(source)(np.concatenate([h.reshape(1,-1),[[reference/10-1]]],-1));maxerr=max(maxerr,float(np.max(np.abs(actual-expected))));assert np.allclose(actual,expected,atol=2e-6)
  later=history(rate+1);actual=p(later);assert p.initial_reference==reference
  expected=NumpyPolicy(source)(np.concatenate([later.reshape(1,-1),[[reference/10-1]]],-1));assert np.allclose(actual,expected,atol=2e-6)
  new=NumpyPolicy(source);new(later);assert abs(new.initial_reference-(rate+1))<2e-6
 # The zero residual control returns exactly the reference for any subsequent history.
 c=NumpyPolicy(target);a=c(history(.8));b=c(history(3.));assert np.allclose(a,-.92,atol=2e-6) and np.array_equal(a,b)
 explicit=np.zeros((3,1585),dtype='float32');explicit[:,-1]=[-1,-.92,-.6];assert np.allclose(c(explicit),explicit[:,-1:],atol=2e-6)
 # This zero-last-layer identity is identical in torch and NumPy, but use an actual torch check.
 import torch
 from rl_algorithms import Agent
 agent=Agent(meta['config']).eval();agent.actor.network.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
 with torch.no_grad():expected=agent.act(torch.from_numpy(explicit)).numpy()
 error=float(np.max(np.abs(c(explicit)-expected)));assert error<2e-4
 meta['numpy_torch_max_abs_action_normalized_error']=error;meta['real_history_numerical_check_samples']=0;meta['identity_check_reference_samples']=3
 target.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 checks={'passed':True,'trained_policy_export':meta['parent_export_sha256'],'implicit_explicit_max_error':maxerr,'reference_control_export':meta['export_sha256'],'checks':['trained policy implicit warmup reference equals explicit context','subsequent history never overwrites initial reference','fresh patient policy instance starts a fresh reference','zero residual control returns fixed observed reference','zero residual control NumPy versus torch actual parity'],'clinical_ready':False}
 (ROOT/'results/anchor_export_tests.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
