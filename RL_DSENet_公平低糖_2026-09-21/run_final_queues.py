import subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;deadline=time.time()+7200
while not (R/'configs/final_manifest.json').exists():
 if time.time()>deadline:raise TimeoutError('No frozen final manifest')
 time.sleep(10)
processes=[]
for label,methods in [('a',['hold','bc','fql','iql','ditr']),('b',['ours','td3bc','rebrac','lom','gfp'])]:
 f=(R/'checks'/('final_queue_'+label+'.log')).open('w',buffering=1)
 processes.append((label,subprocess.Popen([str(P/'.venv/bin/python'),str(R/'run_final.py')]+methods,stdout=f,stderr=subprocess.STDOUT),f))
for label,proc,f in processes:
 rc=proc.wait();f.close();assert rc==0,(label,rc)
subprocess.run([str(P/'.venv/bin/python'),str(R/'final_analysis.py')],check=True)
subprocess.run([str(P/'.venv/bin/python'),str(R/'build_report.py')],check=True)
print('FINAL_EVALUATION_AND_REPORT_COMPLETE',flush=True)
