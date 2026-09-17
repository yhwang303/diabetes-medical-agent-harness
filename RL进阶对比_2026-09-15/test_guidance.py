"""Real-checkpoint tests: no future data access, frozen dynamics, nonzero actor influence, zero-weight ablation."""
import json,copy,hashlib
import torch
from rl_algorithms import Agent
from rl_data import Replay,ROOT
from model_guidance import ModelGuidance

def main():
 torch.set_num_threads(2);torch.manual_seed(812);cfg=json.loads((ROOT/'configs/H00_guidance_smoke.json').read_text());guidance=ModelGuidance(cfg['model_guidance']);data=Replay('validation');b=data.batch(torch.arange(64,device='cuda'));agent=Agent(cfg).cuda();agent.optimizers();before=[{k:v.clone() for k,v in m.state_dict().items()} for m in guidance.models]
 action=agent.act(b['state']);action.retain_grad();loss,info=guidance(b['state'],action);loss.backward();assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum()>0
 for model,weights in zip(guidance.models,before):
  assert all(p.grad is None for p in model.parameters());assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
 agent.model_guidance=guidance;agent.update(b,2);assert all(torch.isfinite(p).all() for p in agent.parameters())
 assert guidance.provenance['event_bank']['source_split']=='train'
 # Identical initialization/RNG with zero guidance must be exactly the baseline update.
 zero=copy.deepcopy(cfg);zero['model_guidance']['weight']=0.;a=Agent(zero).cuda();a.optimizers();plain=copy.deepcopy(zero);plain.pop('model_guidance');c=Agent(plain).cuda();c.load_state_dict(a.state_dict());c.optimizers();a.model_guidance=guidance
 rng=torch.cuda.get_rng_state();a.update(b,2);torch.cuda.set_rng_state(rng);c.update(b,2);assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
 out={'passed':True,'checks':['actual graph checkpoints produce finite nonzero actor gradient','patient models are frozen with no parameter gradients or changes','actor-critic update remains finite','event library is train-only','zero guidance coefficient exactly recovers baseline parameter update'],'guidance_provenance':guidance.provenance,'clinical_ready':False};(ROOT/'results/guidance_tests.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='guidance_provenance'},indent=2))
if __name__=='__main__':main()
