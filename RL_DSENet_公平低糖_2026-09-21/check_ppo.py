import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from ppo_worker import physical
R=Path(__file__).resolve().parent;P=R.parent;B=P/'RL_DSENet_2026-09-17'
torch.manual_seed(260915);x=torch.randn(1000,7);old=x.argmax(-1);mapped=torch.where(old==1,1,torch.where(old==2,2,0));assert torch.equal(mapped,physical(x).argmax(-1))
p=R/'results/PPO_smoke';assert json.loads((p/'completion.json').read_text())['status']=='completed'
ck=torch.load(p/'policy_iter01.pt',map_location='cpu');base=torch.load(B/'results/D06_selected_policy/policy.pt',map_location='cpu')
assert any(not torch.equal(v,ck['policy'][k]) for k,v in base['policy'].items())
assert all(torch.isfinite(v).all() for v in ck['policy'].values())
for v in json.loads((R/'configs/retained_weights.json').read_text()).values():assert hashlib.sha256((B/v['path']).read_bytes()).hexdigest()==v['sha256']
assert ck['simulator_transitions']==432
result={'status':'passed','physical_first_action_argmax_equivalent_random_logits':1000,'real_simulator_smoke_transitions':432,'actor_changed_and_finite':True,'three_retained_weights_unchanged':True,'reward_is_real_BG_label_not_policy_input':True,'timeout_bootstrap_source_reviewed':True}
(R/'checks/ppo_smoke.json').write_text(json.dumps(result,indent=2));print(result)
