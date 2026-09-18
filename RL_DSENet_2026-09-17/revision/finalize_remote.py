"""Wait for the already-running queues, verify all evidence, and archive it."""
import hashlib,json,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 cfg=json.loads((ROOT/'configs/comparison_manifest.json').read_text());started=time.time();previous=None
 required={PROJECT/m[k]/'summary.json' for m in cfg['methods'] for k in ('main_folder','native_folder')}
 while True:
  remaining=[]
  for p in required:
   try:
    s=json.loads(p.read_text());assert s['status']=='completed' and s['count']==60
   except (FileNotFoundError,json.JSONDecodeError,AssertionError):remaining.append(p.parent.name)
  if not remaining:break
  for name in ('td3bc','lom','gfp','iql'):
   failure=ROOT/'results'/f'{name}_seed260915'/'failure.json'
   if failure.exists():raise RuntimeError(failure.read_text())
  if sorted(remaining)!=previous:
   previous=sorted(remaining);print(json.dumps({'waiting_for':previous}),flush=True)
  if time.time()-started>7200:raise TimeoutError('Queues did not complete within two hours')
  time.sleep(15)
 subprocess.run([str(PROJECT/'.venv/bin/python'),str(ROOT/'analyze_extended.py')],check=True)
 delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
 files=[]
 for folder in ('results','configs','checks','paper_tables'):
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.extend(ROOT.glob('*.py'));files.extend(ROOT.glob('*.md'))
 # Deliberately excludes the unlicensed LOM author-source audit snapshots.
 logdir=delivery/'job_logs';logdir.mkdir(exist_ok=True)
 for p in (PROJECT/'.autodl-remote/logs').glob('revision-*.log'):
  dest=logdir/p.name;dest.write_bytes(p.read_bytes());files.append(dest)
 entries={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
 manifest=delivery/'evidence_manifest.json';manifest.write_text(json.dumps({'files':entries,'scorer_sha256':sha(PROJECT/'RL_DITR创新_2026-09-16/control_metrics.py'),'unique_new_evaluation_episodes':780,'training_seed':260915},indent=2))
 archive=delivery/'external_baselines_evidence.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(set(files+[manifest])):t.add(p,arcname=str(p.relative_to(ROOT)))
 result={'archive':str(archive.relative_to(PROJECT)),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(entries)+1}
 (delivery/'archive.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
