"""Same weights/data/scenarios: assess four-hour full-path block planning separately."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
check=json.loads((ROOT/'checks/chunk_planning_check.json').read_text());assert check['status']=='passed'
for run,index in [('R05_recursive_sim_stable_policy',24),('H02_prefix_sim_factual',25),('H03_prefix_sim_paired',26)]:
 deadline=time.time()+7200;folder=ROOT/'results'/run
 while not (folder/'completion.json').exists():
  if (folder/'failure.json').exists():raise RuntimeError('Training failed: '+run)
  if time.time()>deadline:raise TimeoutError('Training did not complete: '+run)
  time.sleep(5)
 subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(folder/'policy_last.pt'),'--mode','chunk','--horizon','48','--block-steps','16','--name','E%02d_%s_long_block_dev'%(index,run.split('_')[0])],check=True)
