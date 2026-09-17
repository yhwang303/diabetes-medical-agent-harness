"""Run one dependent experiment after an existing training job releases the GPU."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--after',required=True);ap.add_argument('--config',required=True);args=ap.parse_args();dependency=ROOT/'results'/args.after
print(json.dumps({'event':'waiting_for_training','after':args.after,'next_config':args.config}),flush=True)
begin=time.time()
while not (dependency/'completion.json').exists():
 if (dependency/'failure.json').exists():raise RuntimeError('Dependency failed; inspect before continuing')
 if time.time()-begin>7200:raise TimeoutError('Dependency did not complete within2h')
 time.sleep(10)
state=json.loads((dependency/'completion.json').read_text());assert state['status']=='two_stage_training_completed_control_evaluation_pending'
time.sleep(5)
subprocess.run([sys.executable,str(ROOT/'train_reference.py'),'--config',args.config],check=True)
