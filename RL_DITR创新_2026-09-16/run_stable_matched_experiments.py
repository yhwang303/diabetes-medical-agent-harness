"""One common repaired continuous-policy objective across matched patient models."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT/'check_policy_objective.py')],check=True)
for config in ['R05','H02','H03']:
 print(json.dumps({'event':'stable_matched_training_start','config':config,'training_seed':260915}),flush=True)
 subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',str(ROOT/'configs'/(config+'.json'))],check=True)
