"""Persistent Torch worker; JSON observable history only, no simulator objects."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ditr_model import DITRAgent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode',choices=['actor','beam','chunk'],default='beam');ap.add_argument('--horizon',type=int,default=48);ap.add_argument('--block-steps',type=int,default=16);ap.add_argument('--check',action='store_true');args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 ck=torch.load(args.checkpoint,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 if args.check:
  from ditr_data import Data,tensor,ROOT
  data=Data('validation');pid=next(iter(data.patients));history=tensor(data.batch(pid,np.arange(2)))['state']
  with torch.no_grad():
   start=time.time();act,info=agent.plan_batch(history);batch_seconds=time.time()-start
   reference=[agent.plan(h[None])[0] for h in history]
   np.testing.assert_allclose(act.cpu().numpy(),reference,atol=1e-5,rtol=1e-5)
   modified=history.clone();modified[1,:,0]+=2.;changed,_=agent.plan_batch(modified);assert abs(float(changed[0]-act[0]))<1e-5
   begin=time.time();agent.plan_batch(history.repeat(8,1,1));torch.cuda.synchronize();seconds=time.time()-begin
  result={'batched_vs_scalar_actions_match':True,'changing_other_patient_does_not_change_first':True,'batch2_seconds':batch_seconds,'batch16_seconds':seconds,'actions_u_h':act.cpu().tolist(),'checkpoint':args.checkpoint,'stage':ck['stage'],'step':ck['step'],'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9}
  (ROOT/'checks/batched_planning.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
 print(json.dumps({'ready':True,'stage':ck['stage'],'step':ck['step']}),flush=True)
 for line in sys.stdin:
  try:
   request=json.loads(line);history=np.asarray(request['history'],dtype='float32')
   if history.ndim!=3 or history.shape[1:]!=(72,22) or not np.isfinite(history).all():raise ValueError('Invalid observable history')
   x=torch.from_numpy(history).cuda();start=time.perf_counter()
   with torch.no_grad():
    if args.mode=='beam':actions,info=agent.plan_batch(x)
    elif args.mode=='chunk':
     from chunk_planning import plan_chunks
     actions,info=plan_chunks(agent,x,horizon=args.horizon,block_steps=args.block_steps)
    else:actions=agent.policy.parameters_at(agent.patient.encode(x)[:,-1])[0];info={}
   result=actions.cpu().tolist()
   if not all(0<=a<=20 for a in result):raise ValueError('Invalid action')
   print(json.dumps({'actions_u_h':result,'batch_seconds':time.perf_counter()-start,'planning':info},allow_nan=False),flush=True)
  except Exception as error:print(json.dumps({'error':repr(error)}),flush=True)

if __name__=='__main__':main()
