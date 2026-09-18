"""Measure matched batch latency only after concurrent CUDA evaluation has ended."""
import hashlib
import json
import os
import subprocess
import time
import numpy as np
import torch
from forecast_model import ROOT
from world_model import Patient,OLD
from policy_bounded import Policy,candidate_actions

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text())
    # Refuse a shared GPU benchmark, rather than compare noisy asynchronous job timing.
    listing=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    others=[int(line.strip()) for line in listing.splitlines() if line.strip().isdigit() and int(line.strip())!=os.getpid()]
    assert not others,others
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    wc=torch.load(ROOT/freeze['world']['path'],map_location='cpu');cfg=wc['config']
    model=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();model.load_state_dict(wc['model']);model.requires_grad_(False)
    policy=Policy().cuda().eval();policy.load_state_dict(torch.load(ROOT/freeze['policy']['path'],map_location='cpu')['policy']);policy.requires_grad_(False)
    cases=json.loads((OLD/'action_probe_dynamic/samples.json').read_text());hist=torch.tensor([c['history'] for c in cases[:16]],device='cuda')
    rows=[]
    with torch.no_grad():
        for batch_size in (1,16):
            x=hist[:batch_size];anchor=model.reference_rate(x)
            for mode in ('planner','actor','beam'):
                elapsed=[]
                for iteration in range(120):
                    torch.cuda.synchronize();start=time.perf_counter()
                    z,f=model.encode(x);rate=model.reference_rate(x);ref=model.reference(z,f,rate);plans=candidate_actions(rate,anchor)
                    if mode=='actor':chosen=policy(z,ref,rate,anchor).argmax(-1)
                    else:
                        q=model.utility(model.trajectories(z,f,rate,plans))
                        if mode=='beam':
                            top=policy(z,ref,rate,anchor).topk(3,dim=-1).indices;allow=torch.zeros_like(q,dtype=torch.bool);allow.scatter_(1,top,True);allow[:,0]=True;q=q.masked_fill(~allow,-torch.inf)
                        chosen=q.argmax(-1)
                    plans[torch.arange(len(x),device=x.device),chosen,0].cpu().tolist();torch.cuda.synchronize()
                    duration=(time.perf_counter()-start)*1000
                    if iteration>=20:elapsed.append(duration)
                rows.append({'mode':mode,'batch_size':batch_size,'repeats':len(elapsed),'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95))})
    output={'device':torch.cuda.get_device_name(),'torch':torch.__version__,'threads':4,'no_other_cuda_processes_before_start':True,'scope':'encode, forecast, reference, candidate scoring/selection, CPU action copy; excludes JSON and simulator','rows':rows,'parameters':{'dsenet':sum(p.numel() for p in model.forecast.parameters()),'history_encoder_including_unused_legacy_heads':sum(p.numel() for p in model.history_encoder.parameters()),'reference_adapter':sum(p.numel() for p in model.reference_adapter.parameters()),'response':sum(p.numel() for p in model.response.parameters()),'policy':sum(p.numel() for p in policy.parameters())},'world_sha256':hashlib.sha256((ROOT/freeze['world']['path']).read_bytes()).hexdigest()}
    (ROOT/'checks/inference_latency.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))

if __name__=='__main__':main()
