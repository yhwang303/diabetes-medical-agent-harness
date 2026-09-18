"""Train on explicit reference and same-state intervention arms only."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from world_model import Patient, ROOT, OLD
from paired_data import PairedData

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--name',default='D02_reference_response');ap.add_argument('--forecast',default='results/D01_forecast/best.pt');args=ap.parse_args()
    out=ROOT/'results'/args.name;out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(260915);torch.set_num_threads(4)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    forecast=ROOT/args.forecast;encoder=OLD/'results/H02_prefix_sim_factual/patient_best.pt'
    patient=Patient(forecast,encoder).cuda().eval();data=PairedData('paired_sim_train_temporal')
    a=data.arrays;history=np.concatenate([a['features'][:,0,:72],np.broadcast_to(data.clock,(data.total,72,2))],-1)
    contexts=[];forecasts=[]
    with torch.no_grad():
        for start in range(0,len(history),32):
            z,f=patient.encode(torch.tensor(history[start:start+32],device='cuda'));contexts.append(z);forecasts.append(f)
    z=torch.cat(contexts);f=torch.cat(forecasts);actions=torch.tensor(a['action'],device='cuda');y=torch.tensor(a['target'],device='cuda');mask=torch.tensor(a['mask'],device='cuda')
    rate=patient.reference_rate(torch.tensor(history,device='cuda'))
    assert torch.allclose(actions[:,0][mask[:,0]],rate[:,None].expand(-1,48)[mask[:,0]],atol=1e-5)
    joint=mask[:,1:]&mask[:,0:1]
    optimizer=torch.optim.AdamW([p for p in patient.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    config={'seed':260915,'epochs':args.epochs,'batch_groups':16,'forecast_checkpoint':str(forecast.relative_to(ROOT)),
            'context_checkpoint':str(encoder.relative_to(ROOT.parent)),'forecast_sha256':hashlib.sha256(forecast.read_bytes()).hexdigest(),
            'context_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(),'training_groups':data.total,
            'training_manifest_sha256':hashlib.sha256((OLD/'paired_sim_train_temporal/manifest.json').read_bytes()).hexdigest(),
            'selection':'fixed final epoch; no development selection','loss':'reference factual MSE + paired effect MSE; mmol/L',
            'reference_semantics':'actual reference arm executing constant last recorded delivered basal',
            'reward':'unchanged RL-DITR point status on same authoritative CGM trajectory; no old reward or V head'}
    config['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'train_world.py',ROOT/'world_model.py',OLD/'response_operator.py')}
    (out/'config.json').write_text(json.dumps(config,indent=2));begin=time.time()
    for epoch in range(args.epochs):
        order=np.random.default_rng(260915+epoch).permutation(data.total);losses=[]
        for start in range(0,len(order),16):
            ix=order[start:start+16];ref=patient.reference(z[ix],f[ix],rate[ix]);pred=patient.response(z[ix],actions[ix,1:]-actions[ix,0:1])
            ref_loss=((ref-y[ix,0]).square()*mask[ix,0]).sum()/mask[ix,0].sum()
            effect_loss=((pred-(y[ix,1:]-y[ix,0:1])).square()*joint[ix]).sum()/joint[ix].sum()
            loss=ref_loss+effect_loss
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in patient.parameters() if p.requires_grad],5.,error_if_nonfinite=True);optimizer.step();losses.append([float(ref_loss),float(effect_loss)])
        if (epoch+1)%10==0:
            event={'epoch':epoch+1,'losses':np.mean(losses,axis=0).tolist(),'seconds':time.time()-begin}
            with (out/'history.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    patient.eval()
    with torch.no_grad():
        same=rate[:,None,None].expand(-1,1,48)
        assert torch.equal(patient.response(z,same-rate[:,None,None]),torch.zeros_like(same))
        changed=same.clone();changed[:,:,24:]+=.25
        original=patient.trajectories(z,f,rate,same);updated=patient.trajectories(z,f,rate,changed)
        torch.testing.assert_close(original[:,:,:24],updated[:,:,:24],atol=0,rtol=0)
        assert (updated[:,:,24:]-original[:,:,24:]).abs().max()>0
    # Save complete, independently reconstructible new world model dependency.
    torch.save({'model':patient.state_dict(),'config':config},out/'world.pt')
    (out/'completion.json').write_text(json.dumps({'status':'world_trained_mechanics_passed_decision_gate_pending',
        'epochs':args.epochs,'zero_intervention_exact':True,'future_action_causal_exact':True,
        'reference_arm_actual_actions_verified':True,'elapsed_seconds':time.time()-begin},indent=2))

if __name__=='__main__':main()
