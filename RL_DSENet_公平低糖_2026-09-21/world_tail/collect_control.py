"""Summarize completed development using the unchanged project aggregator."""
import importlib.util
import time
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent.parent
B = P/'RL_DSENet_2026-09-17'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

path = R.parent/'aggregate.py'
spec = importlib.util.spec_from_file_location('unchanged_control_aggregate', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
directory = R.parent/'results/D_world_tail'
start = time.time()
while not (directory/'summary.json').exists():
    if time.time()-start > 1800: raise TimeoutError('D_world_tail control incomplete after 30min; inspect job log')
    time.sleep(5)
old = module.aggregate(B/'results/F11_actor')
new = module.aggregate(directory)
gate = module.joint(new, old)
out = dict(old=old, new=new, joint=gate, scorer_sha256=sha(path),
           exposed_development=True, actor_weights_unchanged=True,
           training_seed=260915, confirmation_opened=False)
(R/'checks/control_comparison.json').write_text(json.dumps(out, indent=2, allow_nan=False))
print(json.dumps({key: {m: v[m] for m in module.METRICS[:10]} for key, v in [('old', old), ('new', new)]}, indent=2), flush=True)
print(json.dumps(gate, indent=2), flush=True)
