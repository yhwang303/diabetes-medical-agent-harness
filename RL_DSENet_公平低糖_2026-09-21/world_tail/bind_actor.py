"""Bind unchanged D06 actor tensors to W01 without retraining or changing scores."""
import os
from common import R, B, sha, torch, json

source = B/'results/D06_selected_policy/policy.pt'
world = R/'results/W01_tail/world.pt'
out = R/'results/W01_tail/policy_old_actor.pt'
assert not out.exists()
ck = torch.load(source, map_location='cpu')
original = {k: v.clone() for k, v in ck['policy'].items()}
ck['config'] = dict(ck['config'])
ck['config']['world_checkpoint'] = os.path.relpath(world, B)
ck['config']['world_sha256'] = sha(world)
ck['config']['binding_only'] = dict(original_policy_sha256=sha(source), actor_retrained=False)
torch.save(ck, out)
again = torch.load(out, map_location='cpu')
assert set(again['policy']) == set(original)
assert all(torch.equal(v, again['policy'][k]) for k, v in original.items())
check = dict(policy_tensors_exact=True, policy_checkpoint_sha256=sha(out), old_policy_sha256=sha(source),
             world_sha256=sha(world), world_checkpoint=ck['config']['world_checkpoint'])
(R/'checks/actor_binding.json').write_text(json.dumps(check, indent=2))
print(json.dumps(check), flush=True)
