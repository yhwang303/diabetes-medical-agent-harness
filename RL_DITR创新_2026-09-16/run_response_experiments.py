"""Single-seed response-model policy training and matched extra-policy control."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'checks/response_patient_mechanics.json').read_text())['status']=='passed'
for code in ['H04','R07']:
 subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',str(ROOT/'configs'/(code+'.json'))],check=True)
for run,index in [('H04_response_patient_rl',30),('R07_H02_extra_policy_control',32)]:
 folder=ROOT/'results'/run
 assert (folder/'completion.json').exists()
 subprocess.run([sys.executable,str(ROOT/'evaluate_dynamics.py'),'--checkpoint',str(folder/'policy_last.pt'),'--name','E%02d_%s_dynamics'%(index,run.split('_')[0])],check=True)
 subprocess.run([sys.executable,str(ROOT/'evaluate_dynamic_actions.py'),'--run',run,'--weight','policy_last.pt'],check=True)
 subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(folder/'policy_last.pt'),'--mode','beam','--batch-size','4','--name','E%02d_%s_beam_dev'%(index+1,run.split('_')[0])],check=True)
for run,index in [('H04_response_patient_rl',34),('R07_H02_extra_policy_control',35)]:
 subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(ROOT/'results'/run/'policy_last.pt'),'--mode','chunk','--horizon','48','--block-steps','16','--batch-size','4','--name','E%02d_%s_long_block_dev'%(index,run.split('_')[0])],check=True)
