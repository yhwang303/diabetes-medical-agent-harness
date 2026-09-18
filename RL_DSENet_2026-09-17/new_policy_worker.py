"""Observable-history JSON worker; one world for all learned/planning controls."""
import argparse
import json
import time
import numpy as np
import torch
from world_model import Patient,ROOT
from policy import Policy,candidate_actions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode',choices=['actor','planner','beam'],required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    wc=torch.load(ROOT/'results/D02_reference_response/world.pt',map_location='cpu');cfg=wc['config']
    model=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();model.load_state_dict(wc['model']);model.requires_grad_(False)
    policy=None
    if args.mode!='planner':
        policy=Policy().cuda().eval();policy.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['policy']);policy.requires_grad_(False)
    print(json.dumps({'ready':True,'mode':args.mode}),flush=True)
    for line in __import__('sys').stdin:
        try:
            req=json.loads(line);history=np.asarray(req['history'],dtype='float32')
            if history.ndim!=3 or history.shape[1:]!=(72,22) or not np.isfinite(history).all():raise ValueError('Invalid history')
            x=torch.tensor(history,device='cuda');begin=time.perf_counter()
            with torch.no_grad():
                z,f=model.encode(x);rate=model.reference_rate(x);plans=candidate_actions(rate);reference=model.reference(z,f,rate)
                if args.mode=='actor':chosen=policy(z,reference,rate).argmax(-1)
                else:
                    q=model.utility(model.trajectories(z,f,rate,plans))
                    if args.mode=='beam':
                        top=policy(z,reference,rate).topk(3,dim=-1).indices;allow=torch.zeros_like(q,dtype=torch.bool);allow.scatter_(1,top,True);allow[:,0]=True;q=q.masked_fill(~allow,-torch.inf)
                    chosen=q.argmax(-1)
                actions=plans[torch.arange(len(x),device=x.device),chosen,0].cpu().tolist()
            print(json.dumps({'actions_u_h':actions,'batch_seconds':time.perf_counter()-begin}),flush=True)
        except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)

if __name__=='__main__':main()
