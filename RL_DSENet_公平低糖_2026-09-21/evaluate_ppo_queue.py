import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
for iteration in (2,4,8):
 deadline=time.time()+7200
 while True:
  log=R/'results/PPO_real_rewards/history.jsonl'
  complete=[json.loads(x)['iteration'] for x in log.read_text().splitlines()] if log.exists() else []
  if iteration in complete:break
  if time.time()>deadline:raise TimeoutError('PPO checkpoint did not finish')
  time.sleep(10)
 checkpoint=R/'results/PPO_real_rewards'/('policy_iter%02d.pt'%iteration)
 subprocess.run([str(P/'.venv/bin/python'),str(R/'evaluate.py'),'--mode','actor','--bounded-reference','--checkpoint',str(checkpoint),'--config',str(R/'configs/development.json'),'--name','D_ppo%02d'%iteration,'--batch-size','20'],check=True)
 subprocess.run([str(P/'.venv/bin/python'),str(R/'aggregate.py')],check=True)
print('PPO_DEVELOPMENT_COMPLETE',flush=True)
