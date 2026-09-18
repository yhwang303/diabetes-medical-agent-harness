"""Exact finite-candidate expected-return policy gradient; frozen world model."""
import hashlib
import json
import time
import numpy as np
import torch
from world_model import Patient,ROOT,OLD
from forecast_data import Data,tensor
from policy import Policy,candidate_actions

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    world_folder=ROOT/'results/D02_reference_response'
    assert json.loads((world_folder/'development_diagnostics.json').read_text())['decision_gate_passed']
    assert json.loads((ROOT/'checks/policy_mechanics.json').read_text())['status']=='passed'
    out=ROOT/'results/D03_policy';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(260915);torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    ck=torch.load(world_folder/'world.pt',map_location='cpu');cfg=ck['config']
    patient=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();patient.load_state_dict(ck['model']);patient.requires_grad_(False)
    before=state_hash(patient);policy=Policy().cuda();initial=state_hash(policy);optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=1e-4)
    config={'seed':260915,'epochs':1,'batch_size':256,'lr':3e-4,'kl_beta':.05,
        'objective':'maximize exact sum_a pi(a|h) Q_world(h,a) minus0.05 KL(pi||prior)',
        'prior':[.4,.1,.1,.1,.1,.1,.1],'reward':'same authoritative DSENet CGM point trajectory and RL-DITR status',
        'candidates':'hold actual last basal, or +/-0.25U/h in one of three80min blocks;7 plans,48x5min',
        'no_value_bootstrap':True,'old_policy_inherited':False,'world_sha256':hashlib.sha256((world_folder/'world.pt').read_bytes()).hexdigest(),
        'selection':'fixed final one full epoch; no control checkpoint selection',
        'scope':'model-based finite-horizon policy optimization; not offline historical action imitation or original author discrete-dose trainer'}
    (out/'config.json').write_text(json.dumps(config,indent=2));data=Data('train');seen=0;patients=set();begin=time.time();steps=0
    with (out/'history.jsonl').open('w') as log:
        for batch in data.batches(256,260915,horizon=48):
            patients.add(batch['patient']);seen+=len(batch['state']);x=tensor(batch)['state']
            with torch.no_grad():
                z,f=patient.encode(x);rate=patient.reference_rate(x);actions=candidate_actions(rate)
                reference=patient.reference(z,f,rate);q=patient.utility(patient.trajectories(z,f,rate,actions))
                assert torch.isfinite(q).all()
            logits=policy(z,reference,rate);logp=logits.log_softmax(-1);p=logp.exp();advantage=q-q[:,0:1]
            objective=(p*advantage).sum(-1)-.05*(p*(logp-policy.prior.log())).sum(-1);loss=-objective.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
            if steps%100==0:
                event={'step':steps,'seen':seen,'loss':float(loss),'imagined_gain':float((p*advantage).sum(-1).mean()),'elapsed_seconds':time.time()-begin};log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
    after=state_hash(patient);assert before==after and initial!=state_hash(policy);assert seen==1653421 and len(patients)==225
    torch.save({'policy':policy.state_dict(),'config':config},out/'policy.pt')
    result={'status':'policy_training_complete_control_evaluation_pending','steps':steps,'origins':seen,'patients':len(patients),'world_frozen_exact':before==after,'policy_changed':True,'elapsed_seconds':time.time()-begin}
    (out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
