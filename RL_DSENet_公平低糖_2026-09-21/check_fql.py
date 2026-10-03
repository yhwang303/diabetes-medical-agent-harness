import json
from pathlib import Path
import torch
from fql_corrected import Agent
R=Path(__file__).resolve().parent
cfg=json.loads((R/'configs/fql_corrected.json').read_text());cfg.update(width=16,layers=1,state_dim=8)
torch.manual_seed(260915);a=Agent(cfg);a.optimizers()
b={'state':torch.randn(16,8),'next_state':torch.randn(16,8),'action':torch.rand(16,1)*2-1,'next_action':torch.zeros(16,1),'q_valid':torch.ones(16,1,dtype=torch.bool),'reward':torch.randn(16,1)}
errors=[]
for i in (1,2):
 q=[x.detach().clone() for x in a.q.parameters()];t=[x.detach().clone() for x in a.q_target.parameters()]
 a.update(b,i)
 for old,prior,actual in zip(q,t,a.q_target.parameters()):
  expected=prior.lerp(old,cfg['tau']);assert torch.equal(expected,actual);errors.append(float((expected-actual).abs().max()))
 assert any(not torch.equal(x,y) for x,y in zip(q,a.q.parameters()))
result={'status':'passed','two_step_target_equals_pre_optimizer_Polyak':True,'max_abs_error':max(errors),'critic_changed_after_optimizer':True}
(R/'checks/fql_target_order.json').write_text(json.dumps(result,indent=2));print(result)
