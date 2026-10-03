import json
from pathlib import Path
import torch
from rebrac_corrected import Agent
R=Path(__file__).resolve().parent
c=json.loads((R/'configs/rebrac_corrected.json').read_text());c.update(width=16,layers=1,state_dim=8)
torch.manual_seed(260915);a=Agent(c);a.optimizers()
b={'state':torch.randn(16,8),'next_state':torch.randn(16,8),'action':torch.rand(16,1)*2-1,'next_action':torch.zeros(16,1),'q_valid':torch.ones(16,1,dtype=torch.bool),'reward':torch.randn(16,1)}
for step in range(1,5):
 old=[p.detach().clone() for p in a.actor.parameters()];ta=[p.detach().clone() for p in a.actor_target.parameters()];tq=[p.detach().clone() for p in a.q_target.parameters()]
 a.update(b,step)
 if not step%2:
  assert all(torch.equal(x,y) for x,y in zip(ta,a.actor_target.parameters()))
  assert all(torch.equal(x,y) for x,y in zip(tq,a.q_target.parameters()))
 else:
  assert all(torch.equal(t.lerp(o,c['tau']),n) for t,o,n in zip(ta,old,a.actor_target.parameters()))
  assert all(torch.equal(t.lerp(q,c['tau']),n) for t,q,n in zip(tq,a.q.parameters(),a.q_target.parameters()))
  assert any(not torch.equal(x,y) for x,y in zip(old,a.actor.parameters()))
result={'status':'passed','actor_target_from_old_actor':True,'critic_target_from_new_critic':True,'delayed_updates_unchanged':True}
(R/'checks/rebrac_target_order.json').write_text(json.dumps(result,indent=2));print(result)
