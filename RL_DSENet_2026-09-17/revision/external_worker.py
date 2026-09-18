"""Batched inference for frozen new baseline weights; per-scenario noise streams."""
import argparse,json,time
import numpy as np
import torch
from train_baseline import make_agent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode');args=ap.parse_args()
    torch.set_num_threads(2);ck=torch.load(args.checkpoint,map_location='cpu');agent=make_agent(ck['config']).cuda().eval();agent.load_state_dict(ck['agent']);rngs={}
    print(json.dumps({'ready':True,'algorithm':ck['config']['algorithm'],'step':ck['step']}),flush=True)
    for line in __import__('sys').stdin:
        try:
            req=json.loads(line);begin=time.perf_counter();noise=[]
            for key,seed in zip(req['case_keys'],req['scenario_seeds']):
                if key not in rngs:rngs[key]=np.random.default_rng(seed+20000)
                noise.append(rngs[key].normal())
            with torch.no_grad():
                state=torch.tensor(req['history'],dtype=torch.float32,device='cuda').flatten(1)
                action=10*(agent.act(state,torch.tensor(noise,dtype=torch.float32,device='cuda')[:,None])+1)
            actions=action.flatten().cpu().tolist()
            assert all(np.isfinite(x) and -.00001<=x<=20.00001 for x in actions)
            print(json.dumps({'actions_u_h':np.clip(actions,0,20).tolist(),'batch_seconds':time.perf_counter()-begin}),flush=True)
        except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
if __name__=='__main__':main()
