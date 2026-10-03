import hashlib,json,shutil,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;out=R/'delivery';out.mkdir(exist_ok=True)
assert (R/'analysis/final.json').exists() and (out/'最终公平对比与低血糖修复报告.html').exists()
logs=R/'checks/remote_job_logs';logs.mkdir(exist_ok=True)
for p in (P/'.autodl-remote/logs').glob('*.log'):
 if p.name.startswith(('fair-','ditr-shared','world-tail')):shutil.copy2(p,logs/p.name)
# Dependencies stay under their project-relative paths in an independent source archive.
deps=[]
for root in ['RL_DSENet_2026-09-17','RL_DITR创新_2026-09-16','RL进阶对比_2026-09-15','Loop数据集/训练管线_v2']:
 for p in (P/root).rglob('*.py'):
  if not any(k in p.parts for k in ('results','__pycache__','.venv','.git')):deps.append(p)
code=list(R.glob('*.py'))+list((R/'world_tail').glob('*.py'))+list((R/'configs').glob('*.json'))+list(R.glob('*.md'))+deps
with tarfile.open(out/'本轮代码与依赖.tar.gz','w:gz',compresslevel=5) as t:
 for p in sorted(set(code)):t.add(p,arcname=str(p.relative_to(P)),recursive=False)
files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.gz','.tmp') and p.name not in ('归档清单.json','确认评测与报告归档.sha256')]
manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(out/'归档清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(out/'归档清单.json')
trained=json.loads((out/'已完成训练归档清单.json').read_text())
files=[p for p in files if str(p.relative_to(P)) not in trained or hashlib.sha256(p.read_bytes()).hexdigest()!=trained[str(p.relative_to(P))]['sha256']]
archive=out/'确认评测与报告归档.tar.gz';start=time.time()
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p in files:t.add(p,arcname=str(p.relative_to(P)),recursive=False)
actual=hashlib.sha256(archive.read_bytes()).hexdigest();(out/'确认评测与报告归档.sha256').write_text(actual+'  '+archive.name+'\n');print(json.dumps({'files':len(files),'archive_bytes':archive.stat().st_size,'sha256':actual,'seconds':time.time()-start}),flush=True)
