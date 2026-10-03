"""Archive already completed training/diagnostics while confirmation continues."""
import hashlib,json,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;out=R/'delivery';out.mkdir(exist_ok=True)
assert (R/'configs/final_manifest.json').exists()
roots=[p for p in (R/'results').iterdir() if p.is_dir() and not p.name.startswith('C_')]+[R/'shared_replay',R/'world_tail']
files=sorted({p for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp'})
manifest={str(p.relative_to(P)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
mp=out/'已完成训练归档清单.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(mp)
p=out/'已完成训练与开发证据.tar.gz';start=time.time()
with tarfile.open(p,'w:gz',compresslevel=3) as t:
 for f in files:t.add(f,arcname=str(f.relative_to(P)),recursive=False)
sha=hashlib.sha256(p.read_bytes()).hexdigest();(out/'已完成训练与开发证据.sha256').write_text(sha+'  '+p.name+'\n');print(json.dumps({'files':len(files),'bytes':p.stat().st_size,'sha256':sha,'seconds':time.time()-start}),flush=True)
