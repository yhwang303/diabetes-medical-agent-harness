import json
import torch
from world_model import ROOT, OLD, Patient
from policy_bounded import Policy, candidate_actions, history_anchor

def main():
    torch.manual_seed(260915)
    cases=json.loads((OLD/'action_probe_dynamic/samples.json').read_text())
    x=torch.tensor([c['history'] for c in cases],device='cuda')
    rate=Patient.reference_rate(x);anchor=history_anchor(x);plans=candidate_actions(rate,anchor)
    assert plans.shape==(len(x),7,48)
    assert (plans-anchor[:,None,None]).abs().max()<=.25001
    # Arbitrarily many replans cannot move the persistent reference.
    current=rate.clone()
    for _ in range(100):current=candidate_actions(current,anchor)[:,2,0]
    torch.testing.assert_close(current,(anchor+.25).clamp(0,20),atol=0,rtol=0)
    policy=Policy().cuda();z=torch.randn(len(x),256,device='cuda');g=torch.randn(len(x),48,device='cuda')
    logits=policy(z,g,rate,anchor);loss=(logits.softmax(-1)*torch.randn_like(logits)).sum();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    result={'status':'passed','persistent_reference_no_compounding':True,'candidate_support':'+/-0.25 U/h of observed fixed reference','finite_policy_gradients':True,'scope':'research candidate contract, not clinical safety rule'}
    (ROOT/'checks/bounded_policy_mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
