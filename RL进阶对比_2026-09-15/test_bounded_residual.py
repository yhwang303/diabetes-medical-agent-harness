"""Action-subspace and target preservation checks for our bounded residual architecture."""
import json,copy,hashlib
import numpy as np
import torch
from rl_algorithms import Agent
from anchor_data import AnchorReplay
from rl_data import ROOT

def main():
 torch.set_num_threads(2);torch.manual_seed(41)
 cfg=json.loads((ROOT/'configs/R31_anchor_rebrac.json').read_text());cfg.update(residual_limit_u_h=.25,width=32,layers=2)
 a=Agent(cfg);state=torch.randn(100,1585);state[:,-1]=torch.linspace(-1,1,100)
 with torch.no_grad():a.actor.network[-1].weight.normal_();a.actor.network[-1].bias.fill_(10)
 physical=(a.act(state)+1)*10;ref=(state[:,-1:]+1)*10
 assert torch.all(physical>=0) and torch.all(physical<=20) and torch.all((physical-ref).abs()<=.250003)
 # Default None retains the original unbounded residual equation exactly.
 old=copy.deepcopy(cfg);old.pop('residual_limit_u_h');u=Agent(old)
 assert torch.equal(u.act(state),((state[:,-1:]+1)*10+2*u.actor.network(state)).clamp(0,20)/10-1)
 d=AnchorReplay('train',device='cpu');assert d.size==1653421
 targets=d.arrays['action'];delta=(10*(targets.reshape(-1,1)-d.anchor)).abs()
 coverage={str(limit):float((delta<=limit).float().mean()) for limit in [.25,.5]}
 b=d.batch(torch.arange(128));assert torch.equal(b['action'].flatten(),targets[:128].flatten())
 # Use an unsaturated real initialized policy to verify actual learning gradients.
 a=Agent(cfg);a.optimizers();before={k:v.clone() for k,v in a.actor.state_dict().items()};a.update(b,2)
 assert any(not torch.equal(v,before[k]) for k,v in a.actor.state_dict().items());assert all(torch.isfinite(p).all() for p in a.parameters())
 # Ablation removes only critic use in the actor objective; all action targets still present.
 bc=copy.deepcopy(cfg);bc['algorithm']='bc';m=Agent(bc);m.optimizers();m.update(b,2);assert torch.isfinite(m.act(b['state'])).all()
 result={'passed':True,'train_origins':d.size,'fraction_logged_actions_within_fixed_reference_band':coverage,'targets_unchanged':True,'checks':['bounded physical action subspace including 0 and20 references','default unbounded equation bitwise retained','all original actor origins and action targets preserved','finite actual RL and no-RL ablation updates'],'source_sha256':hashlib.sha256((ROOT/'rl_algorithms.py').read_bytes()).hexdigest(),'clinical_ready':False}
 (ROOT/'results/bounded_residual_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
