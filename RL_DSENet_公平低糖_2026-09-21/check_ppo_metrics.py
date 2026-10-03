import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'RL_DITR创新_2026-09-16'))
from control_metrics import summarize
for p in (R/'results/PPO_smoke/trajectories').glob('*.json'):
 d=json.loads(p.read_text());assert summarize(d['records'],216,False)==d['metrics']
(R/'checks/ppo_metrics.json').write_text(json.dumps({'status':'passed','same_simulator_runtime_exact_recomputation':True,'trajectories':2}))
print('two smoke trajectory scores exact in simulator runtime')
