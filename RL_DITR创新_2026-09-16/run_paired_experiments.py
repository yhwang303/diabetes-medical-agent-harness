"""Wait for complete data, verify actual mixed gradients, then train matched variants."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
deadline=time.time()+3600
while not (ROOT/'paired_sim_train/manifest.json').exists():
 if time.time()>deadline:raise TimeoutError('Paired data manifest not complete; no experiment launched')
 time.sleep(5)
subprocess.run([sys.executable,str(ROOT/'check_paired.py')],check=True)
for config in ['R04','H02','H03']:
 print(json.dumps({'event':'matched_training_start','config':config,'training_seed':260915}),flush=True)
 subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',str(ROOT/'configs'/(config+'.json'))],check=True)
