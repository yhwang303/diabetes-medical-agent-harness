"""Locate CUDA training bottlenecks without changing a candidate or score."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from world_model_v2 import WorldModelV2,load_frozen_dsenet

R=Path(__file__).resolve().parent
P=R.parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['deterministic','default','shift'],required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(260915)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=a.mode=='deterministic'
    data=np.load(next((R/'cache/smoke_paired_train').glob('*.npz')))
    h=torch.tensor(np.repeat(data['history'][:1],32,0),device='cuda')
    anchor=torch.tensor(np.repeat(data['anchor_u_h'][:1],32,0),device='cuda')
    u=torch.tensor(np.repeat(data['actions_u_h'][:1],32,0),device='cuda')
    model=WorldModelV2(load_frozen_dsenet(P/'RL_DSENet_2026-09-17/results/P03_g12s3_l2s1/best.pt'),json.loads((P/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text())).cuda().train()
    if a.mode=='shift':
        from types import MethodType
        from torch.nn import functional as F
        def shifted(self,x):
            y=self.norm(x);length=x.shape[1];dilation=self.left_padding//2
            terms=[]
            for j in range(3):
                lag=(2-j)*dilation
                if lag<length:
                    part=F.linear(y[:,:length-lag],self.conv.weight[:,:,j])
                    terms.append(F.pad(part,(0,0,lag,0)))
            result=sum(terms)+self.conv.bias
            value,gate=result.chunk(2,dim=-1)
            return x+self.output(value.tanh()*gate.sigmoid())
        for block in model.action_blocks:block.forward=MethodType(shifted,block)
    for iteration in range(2):
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        cache=model.encode(h,anchor);torch.cuda.synchronize();encoded=time.perf_counter()
        print(json.dumps(dict(mode=a.mode,iteration=iteration,stage='encode',seconds=encoded-start)),flush=True)
        out=model.predict(cache,u);torch.cuda.synchronize();predicted=time.perf_counter()
        print(json.dumps(dict(mode=a.mode,iteration=iteration,stage='predict',seconds=predicted-encoded)),flush=True)
        loss=out['cgm_quantiles_mgdl'].square().mean()/10000+out['bg_event_logits'].square().mean()
        loss.backward();torch.cuda.synchronize();finished=time.perf_counter()
        print(json.dumps(dict(mode=a.mode,iteration=iteration,stage='backward',seconds=finished-predicted,total=finished-start)),flush=True)

if __name__=='__main__':main()
