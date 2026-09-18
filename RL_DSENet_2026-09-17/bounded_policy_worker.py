"""Observable-history JSON worker; one world for all learned/planning controls."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from world_model import Patient,ROOT
from policy_bounded import Policy,candidate_actions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode',choices=['actor','planner','beam'],required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    policy_checkpoint=None
    if args.mode=='planner':world_path=args.checkpoint
    else:
        policy_checkpoint=torch.load(args.checkpoint,map_location='cpu')
        world_path=ROOT/policy_checkpoint['config'].get('world_checkpoint','results/D02_reference_response/world.pt')
        from pathlib import Path
        assert hashlib.sha256(Path(world_path).read_bytes()).hexdigest()==policy_checkpoint['config']['world_sha256']
    wc=torch.load(world_path,map_location='cpu');cfg=wc['config']
    model=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();model.load_state_dict(wc['model']);model.requires_grad_(False)
    policy=None
    if args.mode!='planner':
        policy=Policy().cuda().eval();policy.load_state_dict(policy_checkpoint['policy']);policy.requires_grad_(False)
    print(json.dumps({'ready':True,'mode':args.mode}),flush=True)
    for line in __import__('sys').stdin:
        try:
            req=json.loads(line);history=np.asarray(req['history'],dtype='float32')
            if history.ndim!=3 or history.shape[1:]!=(72,22) or not np.isfinite(history).all():raise ValueError('Invalid history')
            x=torch.tensor(history,device='cuda');begin=time.perf_counter()
            with torch.no_grad():
                z,f=model.encode(x);rate=model.reference_rate(x);anchor=torch.tensor(req["anchor_u_h"],device=x.device,dtype=x.dtype)
                if anchor.shape!=rate.shape or not torch.isfinite(anchor).all():raise ValueError("Invalid persistent reference")
                if ((rate-anchor).abs()>.26).any():raise ValueError("Current rate outside frozen research candidate support")
                plans=candidate_actions(rate,anchor);reference=model.reference(z,f,rate)
                if args.mode=='actor':chosen=policy(z,reference,rate,anchor).argmax(-1)
                else:
                    q=model.utility(model.trajectories(z,f,rate,plans))
                    if args.mode=='beam':
                        top=policy(z,reference,rate,anchor).topk(3,dim=-1).indices;allow=torch.zeros_like(q,dtype=torch.bool);allow.scatter_(1,top,True);allow[:,0]=True;q=q.masked_fill(~allow,-torch.inf)
                    chosen=q.argmax(-1)
                actions=plans[torch.arange(len(x),device=x.device),chosen,0].cpu().tolist()
            print(json.dumps({'actions_u_h':actions,'batch_seconds':time.perf_counter()-begin}),flush=True)
        except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)

if __name__=='__main__':main()
