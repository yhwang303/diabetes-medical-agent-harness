"""Finite dependency runner; no new optimization or checkpoint overwrite."""
from pathlib import Path
import time, subprocess, sys
root=Path(__file__).resolve().parent
status=root.parent/'.autodl-remote/jobs/rl-final-validation-20260915.status'
for _ in range(240):
 text=status.read_text() if status.exists() else ''
 if 'exit_code=' in text:
  if 'exit_code=0\n' not in text:raise RuntimeError('Full validation dependency failed: '+text)
  break
 time.sleep(15)
else:raise TimeoutError('Full validation dependency did not finish')
for run in ['A02_lower_learning_rate','A03_residual_target']:
 subprocess.run([sys.executable,str(root/'input_diagnostics.py'),'--run',run],check=True)
subprocess.run([sys.executable,str(root/'export_artifacts.py')],check=True)
