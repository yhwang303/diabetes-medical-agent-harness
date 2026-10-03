"""Old/new actor under unchanged candidates; explicit post-suspension re-entry."""
import sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;B=R.parent/'RL_DSENet_2026-09-17';sys.path.insert(0,str(B))
from world_model import Patient
from policy_bounded import Policy,candidate_actions

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode',default='actor');ap.add_argument('--allow-reentry',action='store_true');args=ap.parse_args();assert args.mode=='actor'
 torch.set_num_threads(2);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 ck=torch.load(args.checkpoint,map_location='cpu');wp=B/ck['config']['world_checkpoint'];assert hashlib.sha256(wp.read_bytes()).hexdigest()==ck['config']['world_sha256'];wc=torch.load(wp,map_location='cpu');c=wc['config'];model=Patient(B/c['forecast_checkpoint'],B.parent/c['context_checkpoint']).cuda().eval();model.load_state_dict(wc['model']);model.requires_grad_(False)
 policy=Policy().cuda().eval();policy.load_state_dict(ck['policy']);policy.requires_grad_(False)
 print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  try:
   req=json.loads(line);x=torch.tensor(np.asarray(req['history'],dtype='float32'),device='cuda');assert x.ndim==3 and x.shape[1:]==(72,22) and torch.isfinite(x).all();anchor=torch.tensor(req['anchor_u_h'],device='cuda',dtype=x.dtype);start=time.perf_counter()
   with torch.no_grad():
    rate=model.reference_rate(x);assert anchor.shape==rate.shape and torch.isfinite(anchor).all();within=(rate-anchor).abs()<=.26
    if not args.allow_reentry:assert within.all(),'Outside original candidate support'
    # A previous continuous reduction is outside learned current-rate support. Do not
    # invent its Q. Before invoking the actor again, request one original lower
    # rate step, subject to the same external suspension rule.
    if (~within).any():assert ((rate[~within]>=-.001)&(rate[~within]<=anchor[~within]+.26)).all(),'Unsupported rate must be a logged reduction'
    actions=(anchor-.25).clamp(0,20)
    if within.any():
     z,f=model.encode(x[within]);reference=model.reference(z,f,rate[within]);chosen=policy(z,reference,rate[within],anchor[within]).argmax(-1);plans=candidate_actions(rate[within],anchor[within]);actions[within]=plans[torch.arange(len(chosen),device='cuda'),chosen,0]
   print(json.dumps({'actions_u_h':actions.cpu().tolist(),'reentry':(~within).cpu().tolist(),'batch_seconds':time.perf_counter()-start}),flush=True)
  except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
if __name__=='__main__':main()
