"""Gradient identity and genuine DSENet-to-utility dependency before policy training."""
import json
import argparse
import torch
from world_model import Patient, ROOT, OLD
from policy import Policy, candidate_actions

def main():
    torch.set_num_threads(4);torch.manual_seed(260915);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    ap=argparse.ArgumentParser();ap.add_argument('--world',default='D02_reference_response');args=ap.parse_args()
    folder=ROOT/'results'/args.world
    wc=torch.load(folder/'world.pt',map_location='cpu');cfg=wc['config']
    model=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();model.load_state_dict(wc['model']);model.requires_grad_(False)
    cases=json.loads((OLD/'action_probe_dynamic/samples.json').read_text());x=torch.tensor([c['history'] for c in cases],device='cuda')
    with torch.no_grad():
        z,f=model.encode(x);rate=model.reference_rate(x);plans=candidate_actions(rate)
        y=model.trajectories(z,f,rate,plans);q=model.utility(y)
        for parameter in model.history_encoder.glucose.parameters():parameter.fill_(float('nan'))
        for parameter in model.history_encoder.reward.parameters():parameter.fill_(float('nan'))
        for parameter in model.history_encoder.value.parameters():parameter.fill_(float('nan'))
        z2,f2=model.encode(x);torch.testing.assert_close(y,model.trajectories(z2,f2,rate,plans),atol=0,rtol=0)
        model.forecast.backbone.head.linear.bias.add_(.25)
        _,changed=model.encode(x);qy=model.utility(model.trajectories(z,changed,rate,plans))
        assert (qy-q).abs().max()>1e-4
    logits=torch.randn(3,7,device='cuda',requires_grad=True);values=torch.randn(3,7,device='cuda');prior=Policy().cuda().prior
    p=logits.softmax(-1);logp=logits.log_softmax(-1)
    objective=(p*values).sum(-1)-.05*(p*(logp-prior.log())).sum(-1)
    actual=torch.autograd.grad(objective.sum(),logits)[0]
    centered=values-.05*(logp-prior.log()+1)
    expected=p*(centered-(p*centered).sum(-1,keepdim=True))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    result={'status':'passed','exact_expected_return_gradient':True,'old_glucose_reward_value_heads_unused':True,
            'perturb_DSE_forecast_changes_world_utility':True,'candidate_count':7,'prediction_queries':len(cases),
            'not_control_validation':True}
    (folder/'policy_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
