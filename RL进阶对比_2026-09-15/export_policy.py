"""Exact policy export plus independent NumPy inference numerical check on real histories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from rl_algorithms import Agent
from rl_data import ROOT,Replay
from numpy_policy import NumpyPolicy

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--step',type=int,required=True);args=ap.parse_args()
 torch.set_num_threads(1);torch.manual_seed(417);folder=ROOT/'results'/args.run;source=folder/('policy_%07d.pt'%args.step);ck=torch.load(source,map_location='cpu');agent=Agent(ck['config']).eval();agent.load_state_dict(ck['agent'])
 network=agent.actor.network if ck['config'].get('anchor_context') else agent.actor
 dest=folder/('export_%07d.npz'%args.step);np.savez(dest,**{k:v.detach().numpy() for k,v in network.state_dict().items()})
 modules=[{'key':str(i),'type':type(m).__name__,**({'eps':m.eps} if isinstance(m,torch.nn.LayerNorm) else {})} for i,m in enumerate(network)]
 meta={'config':ck['config'],'step':ck['step'],'modules':modules,'output_transform':'anchor_plus_2_residual' if ck['config'].get('anchor_context') else None,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'clinical_ready':False};dest.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 DataClass=Replay
 if ck['config'].get('anchor_context'):
  from anchor_data import AnchorReplay
  DataClass=AnchorReplay
 replay=DataClass('validation',device='cpu');indices=torch.randperm(replay.size)[:256];state=replay.batch(indices)['state'];noise=torch.randn(256,1)
 with torch.no_grad():expected=agent.act(state,noise).numpy()
 actual=NumpyPolicy(dest)(state.numpy(),noise.numpy());error=float(np.max(np.abs(expected-actual)));assert error<2e-4,error
 meta['numpy_torch_max_abs_action_normalized_error']=error;meta['real_history_numerical_check_samples']=256;dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['config','modules']},indent=2))

if __name__=='__main__':main()
