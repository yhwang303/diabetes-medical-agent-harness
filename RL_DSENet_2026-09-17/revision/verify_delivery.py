"""Verify archived evidence before extracting data without replacing local sources."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'delivery/archive.json').read_text())
archive=ROOT/'delivery/external_baselines_evidence.tar.gz'
def digest(f):
 h=hashlib.sha256()
 while True:
  b=f.read(8*1024*1024)
  if not b:break
  h.update(b)
 return h.hexdigest()
with archive.open('rb') as f:assert digest(f)==meta['sha256']
assert archive.stat().st_size==meta['bytes']
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();assert len(members)==meta['files']
 assert len({m.name for m in members})==len(members)
 for m in members:
  p=PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts,m.name
 manifest=json.load(t.extractfile('delivery/evidence_manifest.json'))
 assert set(manifest['files'])=={m.name for m in members}-{'delivery/evidence_manifest.json'}
 for m in members:
  if m.name in manifest['files']:assert digest(t.extractfile(m))==manifest['files'][m.name],m.name
 extracted=[]
 for m in members:
  if PurePosixPath(m.name).parts[0] in ('results','checks','paper_tables','configs','delivery'):
   dest=ROOT/m.name
   if dest.exists():
    with dest.open('rb') as f:assert digest(f)==digest(t.extractfile(m)),f'Existing different file: {m.name}'
   else:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(m) as src,dest.open('wb') as dst:
     while True:
      b=src.read(8*1024*1024)
      if not b:break
      dst.write(b)
   extracted.append(m.name)
result={'archive_sha256':meta['sha256'],'archive_bytes':meta['bytes'],'archive_files':len(members),'file_hashes_verified':len(manifest['files']),'data_files_extracted_or_matched':len(extracted),'source_files_preserved_locally':True,'all_archive_paths_safe_regular_files':True}
(ROOT/'checks/local_delivery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
