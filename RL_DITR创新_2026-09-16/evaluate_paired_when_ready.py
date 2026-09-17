"""Evaluate each matched experiment when complete; training outcomes stay separate."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for run,index in [('R05_recursive_sim_stable_policy',17),('H02_prefix_sim_factual',19),('H03_prefix_sim_paired',21)]:
 deadline=time.time()+7200;folder=ROOT/'results'/run
 while not (folder/'completion.json').exists():
  if (folder/'failure.json').exists():raise RuntimeError('Training failed: '+run)
  if time.time()>deadline:raise TimeoutError('Training did not complete: '+run)
  time.sleep(5)
 print(json.dumps({'event':'paired_experiment_evaluation','run':run}),flush=True)
 subprocess.run([sys.executable,str(ROOT/'evaluate_dynamics.py'),'--checkpoint',str(folder/'policy_last.pt'),'--name','E%02d_%s_dynamics'%(index,run.split('_')[0])],check=True)
 subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(folder/'policy_last.pt'),'--mode','beam','--name','E%02d_%s_final_beam_dev'%(index+1,run.split('_')[0])],check=True)
