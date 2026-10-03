"""CUDA mechanics checks with real frozen DSENet and simulator histories."""
import os
os.environ['OMP_NUM_THREADS']='2'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from world_model_v2 import WorldModelV2,load_frozen_dsenet

R=Path(__file__).resolve().parent
P=R.parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='paired_smoke_r1');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(260915)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    paths=sorted((R/'cache'/a.data).glob('*.npz'))
    assert len(paths)>=2
    packs=[np.load(p) for p in paths[:2]]
    h=torch.tensor(np.concatenate([p['history'][:1] for p in packs]),device='cuda')
    anchor=torch.tensor(np.concatenate([p['anchor_u_h'][:1] for p in packs]),device='cuda')
    actions=torch.tensor(np.concatenate([p['actions_u_h'][:1] for p in packs]),device='cuda')
    path=P/'RL_DSENet_2026-09-17/results/P03_g12s3_l2s1/best.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='c01610104632a967b4de5eaafe59511a67e3b4d0d34954fab5bc7341fb8d19b4'
    forecast=load_frozen_dsenet(path).cuda()
    model=WorldModelV2(forecast,json.loads((P/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text())).cuda().eval()
    errors={}
    with torch.no_grad():
        cache=model.encode(h,anchor);out=model.predict(cache,actions)
        errors['real_dsenet_mgdl']=float((cache['dsenet_cgm_4h_mgdl']-forecast(h)*18).abs().max())
        assert errors['real_dsenet_mgdl']==0
        assert out['cgm_quantiles_mgdl'].shape==(2,9,72,7)
        assert (out['cgm_quantiles_mgdl'].diff(dim=-1)>0).all()
        assert (out['p_bg_below54']<=out['p_bg_below70']).all()
        altered=actions.clone();altered[:,:,24:]=anchor[:,None,None]*1.8
        change=model.predict(cache,altered)
        errors['future_actions_prefix']=float((out['cgm_quantiles_mgdl'][:,:,:24]-change['cgm_quantiles_mgdl'][:,:,:24]).abs().max())
        assert errors['future_actions_prefix']==0
        shifted=h.clone();shifted[:,:,:5]=torch.where(h[:,:,5:10]>.5,h[:,:,:5],12345.)
        masked=model(shifted,anchor,actions)
        errors['masked_placeholder']=float((out['cgm_quantiles_mgdl']-masked['cgm_quantiles_mgdl']).abs().max())
        assert errors['masked_placeholder']==0
        single=model(h[:1],anchor[:1],actions[:1])
        errors['batch_single']=float((out['cgm_quantiles_mgdl'][:1]-single['cgm_quantiles_mgdl']).abs().max())
        assert errors['batch_single']<1e-3
        for horizon in (1,48,49,72):
            short=model.predict(cache,actions[:,:,:horizon])
            assert short['cgm_quantiles_mgdl'].shape==(2,9,horizon,7)
            assert short['dsenet_available'].sum()==min(horizon,48)
            errors['horizon_%d_prefix'%horizon]=float((short['cgm_quantiles_mgdl']-out['cgm_quantiles_mgdl'][:,:,:horizon]).abs().max())
            assert errors['horizon_%d_prefix'%horizon]<1e-3
    model.train();grad_actions=actions.clone().requires_grad_(True)
    pred=model(h,anchor,grad_actions)
    loss=pred['cgm_median_mgdl'].mean()+pred['bg_event_logits'].square().mean()
    loss.backward()
    grads={name:float(sum((p.grad.abs().sum().item() if p.grad is not None else 0) for p in module.parameters()))
           for name,module in [('forecast',model.forecast),('history',model.history_encoder),('actions',model.future_input)]}
    grads['action_input']=float(grad_actions.grad.abs().sum())
    assert grads['forecast']==0 and min(grads[k] for k in ('history','actions','action_input'))>0
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize();begin=time.perf_counter()
        for _ in range(20):model(h,anchor,actions)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)/20
    result=dict(status='passed',data=a.data,errors=errors,gradient_sums=grads,
                seconds_per_2_histories_18_arms=elapsed,peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'world_model_v2.py']},
                performance_or_calibration_claim=False)
    dest=R/'checks/world_mechanics.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
