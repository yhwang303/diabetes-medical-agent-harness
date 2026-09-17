"""Matched predeclared diagnostic modes; no new model training or baseline reruns."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
jobs=[('R03_reference_categorical','policy_last.pt','actor','E09_R03_final_actor_dev'),('H01_action_prefix','policy_last.pt','actor','E10_H01_final_actor_dev'),('R03_reference_categorical','policy_002000.pt','beam','E11_R03_policy2k_beam_dev'),('H01_action_prefix','policy_002000.pt','beam','E12_H01_policy2k_beam_dev')]
for folder,weight,mode,name in jobs:
 print(json.dumps({'event':'evaluation_start','name':name}),flush=True)
 subprocess.run([sys.executable,str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/development_control.json'),'--checkpoint',str(ROOT/'results'/folder/weight),'--mode',mode,'--name',name],check=True)
