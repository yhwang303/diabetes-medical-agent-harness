"""Matched delayed-objective experiments; final epoch fixed before evaluation."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'checks/delayed_policy_mechanics.json').read_text())['status']=='passed'
for code in ['H05','R08']:
 subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',str(ROOT/'configs'/(code+'.json'))],check=True)
for code,index in [('H05',39),('R08',41)]:
 cfg=json.loads((ROOT/'configs'/(code+'.json')).read_text());folder=ROOT/'results'/cfg['name']
 completion=json.loads((folder/'policy_completion.json').read_text())
 assert completion['full_epochs']==1 and completion['samples_seen']==1653421 and completion['patients_seen']==225
 for mode,offset in [('beam',0),('chunk',1)]:
  command=[str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(folder/'policy_last.pt'),'--mode',mode,'--batch-size','4','--name','E%02d_%s_%s_dev'%(index+offset,code,mode)]
  if mode=='chunk':command+=['--horizon','48','--block-steps','16']
  subprocess.run(command,check=True)
