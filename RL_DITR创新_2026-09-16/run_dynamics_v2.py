"""Recompute existing probes with delivered dose conditioning; no new simulation."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for folder,name in [('R01b_reference_math_eval_resume','E13_R01_dynamics_delivered'),('R03_reference_categorical','E14_R03_dynamics_delivered'),('H01_action_prefix','E15_H01_dynamics_delivered')]:
 subprocess.run([sys.executable,str(ROOT/'evaluate_dynamics.py'),'--checkpoint',str(ROOT/'results'/folder/'patient_best.pt'),'--name',name],check=True)
