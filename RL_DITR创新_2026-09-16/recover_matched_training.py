"""Retry zero-update infrastructure failure, then serialize large planning work."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
failure=json.loads((ROOT/'results/H02_prefix_sim_factual_oom_before_update/failure.json').read_text())
assert failure['step']==0 and 'OutOfMemoryError' in failure['error']
for code in ['H02','H03']:
 print(json.dumps({'event':'matched_training','config':code,'training_seed':260915}),flush=True)
 subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',str(ROOT/'configs'/(code+'.json'))],check=True)
for run,index in [('R05_recursive_sim_stable_policy',29),('H02_prefix_sim_factual',25),('H03_prefix_sim_paired',26)]:
 folder=ROOT/'results'/run
 assert (folder/'completion.json').exists()
 subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(folder/'policy_last.pt'),'--mode','chunk','--horizon','48','--block-steps','16','--batch-size','4','--name','E%02d_%s_long_block_dev'%(index,run.split('_')[0])],check=True)
