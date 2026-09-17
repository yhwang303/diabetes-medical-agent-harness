"""Wait for existing weights/data, then run common timing diagnostics without retraining."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def wait_for(path,failure=None):
 deadline=time.time()+7200
 while not path.exists():
  if failure and failure.exists():raise RuntimeError(failure.read_text())
  if time.time()>deadline:raise TimeoutError(str(path))
  time.sleep(5)
wait_for(ROOT/'action_probe_dynamic/manifest.json')
for run,weight in [('R05_recursive_sim_stable_policy','policy_last.pt'),('H02_prefix_sim_factual','patient_best.pt'),('H03_prefix_sim_paired','patient_best.pt')]:
 folder=ROOT/'results'/run
 wait_for(folder/('completion.json' if run.startswith('R05') else 'patient_completion.json'),folder/'failure.json')
 subprocess.run([sys.executable,str(ROOT/'evaluate_dynamic_actions.py'),'--run',run,'--weight',weight],check=True)
 if run.startswith('H03'):subprocess.run([sys.executable,str(ROOT/'diagnose_action_identification.py'),'--run',run],check=True)
