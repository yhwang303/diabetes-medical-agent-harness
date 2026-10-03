import json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'RL进阶对比_2026-09-15'))
from rl_data import Replay
from lom_corrected import Agent
c=json.loads((R/'configs/lom_corrected.json').read_text());c.update(gmm_pretrain_steps=2)
torch.set_num_threads(2);torch.manual_seed(c['seed']);a=Agent(c).cuda();a.optimizers();r=Replay('validation');b=r.batch(torch.arange(32,device='cuda'))
for step in range(1,7):
 old=[p.detach().clone() for p in a.q_target.parameters()];info=a.update(b,step)
 assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
 if step>2 and (step-3)%2==0:
  assert all(torch.equal(o.lerp(q,c['tau']),t) for o,q,t in zip(old,a.q.parameters(),a.q_target.parameters()))
  assert any(not torch.equal(o,t) for o,t in zip(old,a.q_target.parameters()))
 else:assert all(torch.equal(o,t) for o,t in zip(old,a.q_target.parameters()))
with torch.no_grad():assert torch.isfinite(a.act(b['state'])).all()
result={'status':'passed','real_Loop_CUDA_rows':32,'target_phase_first_third_RL_step':True,'finite_training':True}
(R/'checks/lom_target_order.json').write_text(json.dumps(result,indent=2));print(result)
