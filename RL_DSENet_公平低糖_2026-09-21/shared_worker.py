import argparse,json,sys,time
import numpy as np
import torch
from train_shared import agent_class
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);args=ap.parse_args()
torch.set_num_threads(2);ck=torch.load(args.checkpoint,map_location='cpu');c=ck['config'];agent=agent_class(c['algorithm'])(c).cuda().eval();agent.load_state_dict(ck['agent']);rngs={}
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line);start=time.perf_counter();noise=[]
  for key,seed in zip(req['case_keys'],req['scenario_seeds']):
   if key not in rngs:rngs[key]=np.random.default_rng(seed+20000)
   noise.append(rngs[key].normal())
  with torch.no_grad():
   x=torch.tensor(req['history'],dtype=torch.float32,device='cuda').flatten(1);z=torch.tensor(noise,dtype=torch.float32,device='cuda')[:,None];u=10*(agent.act(x,z)+1)
  assert torch.isfinite(u).all();actions=np.clip(u.flatten().cpu().numpy(),0,20).tolist()
  print(json.dumps({'actions_u_h':actions,'batch_seconds':time.perf_counter()-start}),flush=True)
 except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
