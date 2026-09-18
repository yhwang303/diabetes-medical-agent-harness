"""Gradient and real-data CUDA smoke before any full-budget run."""
import copy,json,sys
from pathlib import Path
import torch
from train_baseline import ROOT,Replay,make_agent

torch.set_num_threads(2);torch.manual_seed(260915)
data=Replay('validation');b=data.batch(torch.arange(64,device='cuda'))
checks={}
for name in ('gfp','td3bc','iql','lom'):
 cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
 if name=='lom':cfg['gmm_pretrain_steps']=2
 agent=make_agent(cfg).cuda();agent.optimizers()
 before={k:v.clone() for k,v in agent.state_dict().items()}
 for step in range(1,7):
  info=agent.update(b,step)
  assert all(torch.isfinite(torch.tensor(v)) for v in info.values()),(name,info)
 with torch.no_grad():
  actions=agent.act(b['state']);assert actions.shape==(64,1) and torch.isfinite(actions).all() and (actions.abs()<=1).all()
 assert any(not torch.equal(v,agent.state_dict()[k]) for k,v in before.items())
 assert any(not torch.equal(v,agent.state_dict()[k]) for k,v in before.items() if k.startswith('actor.'))
 checkpoint={'config':cfg,'agent':agent.state_dict()};restored=make_agent(cfg).cuda();restored.load_state_dict(checkpoint['agent'])
 noise=torch.zeros(64,1,device='cuda')
 with torch.no_grad():assert torch.equal(agent.act(b['state'],noise),restored.act(b['state'],noise))
 if name=='gfp':
  # Equal dataset and reference actions must give exactly 1/2 guidance.
  weight=agent.guidance(b['state'],b['action'],b['action'],torch.tensor(1.,device='cuda'))
  assert torch.equal(weight,torch.full_like(weight,.5))
 checks[name]={'status':'passed','real_rows':64,'finite_updates':6,'actor_changed':True,'bounded':True,'save_reload_exact':True,'last_diagnostic':info}
 del agent,restored;torch.cuda.empty_cache()
write=ROOT/'checks';write.mkdir(exist_ok=True);(write/'baseline_smoke.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
