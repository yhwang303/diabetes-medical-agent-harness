"""Archive complete raw evidence only after all frozen evaluation queues exit."""
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text());begin=time.time()
    names=['dsenet-final-'+queue+'-20260917' for queue in ('baselines','ditr','ours')]
    while True:
        complete=True
        for name in names:
            status=ROOT.parent/'.autodl-remote/jobs'/(name+'.status')
            text=status.read_text() if status.exists() else ''
            if 'status=exited' in text and 'exit_code=0' not in text:raise RuntimeError('Final queue failed: '+name+' '+text)
            complete &= 'status=exited' in text and 'exit_code=0' in text
        if complete:break
        if time.time()-begin>5400:raise TimeoutError('Final evaluation queues did not complete')
        time.sleep(30)
    for item in freeze['methods']:assert (ROOT/'results'/item['name']/'summary.json').exists()
    assert (ROOT/'results/F00_forecast_heldout/evaluation.json').exists()
    subprocess.run([str(ROOT.parent/'.venv/bin/python'),str(ROOT/'analyze_control.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'benchmark_frozen.py')],check=True)
    files=[]
    for name in ('results','checks','configs','paper_tables'):
        files += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    logdir=ROOT/'delivery/job_logs';logdir.mkdir(exist_ok=True)
    import shutil
    for path in (ROOT.parent/'.autodl-remote/logs').glob('dsenet-*.log'):
        shutil.copy2(path,logdir/path.name)
    files+=list(logdir.glob('*.log'))
    manifest={'status':'all_frozen_evaluations_complete','freeze_sha256':hashlib.sha256((ROOT/'configs/final_freeze.json').read_bytes()).hexdigest(),'methods':len(freeze['methods']),'control_trajectories':len(freeze['methods'])*freeze['episode_count'],'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    path=ROOT/'delivery/final_evidence_manifest.json';path.write_text(json.dumps(manifest,indent=2));files.append(path)
    with tarfile.open(ROOT/'delivery/final_evidence.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)))
    print(json.dumps({'status':'final_evidence_archived','files':len(files),'bytes':(ROOT/'delivery/final_evidence.tar.gz').stat().st_size}),flush=True)

if __name__=='__main__':main()
