from pathlib import Path
import os,json,hashlib,zipfile,zlib
R=Path(__file__).resolve().parents[3]; N=R/'Loop数据集'; O=N/'说明文档/核验资料'
moves=[('datasets/raw/loop','Loop数据集/官方压缩包'),('datasets/extracted/loop','Loop数据集/原始数据'),('datasets/audit/2026-09-12/loop','Loop数据集/历史资料/2026-09-12审计'),('datasets/manifests/loop_download.json','Loop数据集/历史资料/获取记录/loop_download.json'),('datasets/private/loop-submission.json','Loop数据集/历史资料/私密申请记录/loop-submission.json')]
for date,name in [('2026-09-11','loop-download-form.html'),('2026-09-12','loop-form.html'),('2026-09-12','loop-transfer.log')]:
 moves.append((f'datasets/access/{date}/{name}',f'Loop数据集/历史资料/获取记录/{date}/{name}'))
if (O/'迁移清单.json').exists(): raise RuntimeError('Migration already recorded; do not repeat')
allrecords=[]
for old,new in moves:
 src=R/old;dst=R/new
 if not src.exists() or dst.exists():raise RuntimeError(f'Unexpected source/destination: {old}')
 for f in ([src] if src.is_file() else sorted(src.rglob('*'))):
  if f.is_symlink():raise RuntimeError('Unexpected symlink')
  if not f.is_file():continue
  rel=Path() if src.is_file() else f.relative_to(src)
  st=f.stat();record={'old':str(f.relative_to(R)),'new':str((dst/rel).relative_to(R)),'size':st.st_size,'inode':st.st_ino,'device':st.st_dev,'mtime_ns':st.st_mtime_ns,'mode':oct(st.st_mode&0o777)}
  if st.st_size<10_000_000:record['sha256_before']=hashlib.sha256(f.read_bytes()).hexdigest()
  allrecords.append(record)
(O/'迁移前清单.json').write_text(json.dumps(allrecords,ensure_ascii=False,indent=2))
for old,new in moves:
 src=R/old;dst=R/new;dst.parent.mkdir(parents=True,exist_ok=True)
 if '私密' in new:dst.parent.chmod(0o700)
 src.rename(dst)
for rec in allrecords:
 f=R/rec['new'];st=f.stat()
 assert (st.st_size,st.st_ino,st.st_dev,st.st_mtime_ns)==(rec['size'],rec['inode'],rec['device'],rec['mtime_ns']),rec['new']
 if 'sha256_before' in rec:assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256_before']
 rec['move_preserved_inode_size_mtime']=True
(O/'迁移清单.json').write_text(json.dumps({'moves':moves,'files':allrecords,'method':'same-filesystem rename; no raw file contents rewritten'},ensure_ascii=False,indent=2))
print(f'Moved {len(allrecords)} files; all inode/size/mtime identities preserved',flush=True)
archive=N/'官方压缩包/Loop study public dataset 2023-01-31.zip'
with archive.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
previous=json.loads((N/'历史资料/获取记录/loop_download.json').read_text())
# Original manifest schema may nest archive metadata; compare the known independently recorded digest.
assert sha=='d63ab91c14d74034a262353a7a82587f65c914849fcac7f6239719f5c69bf4c6',sha
files=[]
with zipfile.ZipFile(archive) as z:
 for info in z.infolist():
  if info.is_dir():continue
  f=N/'原始数据'/info.filename;crc=0;h=hashlib.sha256();size=0
  with f.open('rb') as src:
   while chunk:=src.read(8*1024*1024):size+=len(chunk);crc=zlib.crc32(chunk,crc);h.update(chunk)
  assert size==info.file_size and crc==info.CRC,info.filename
  files.append({'file':info.filename,'bytes':size,'crc32':f'{crc:08x}','sha256':h.hexdigest(),'matches_official_zip_size_crc':True})
  print('verified',info.filename,flush=True)
assert len(files)==47
(O/'原始文件校验.json').write_text(json.dumps({'archive_sha256':sha,'files_count':len(files),'files_bytes':sum(f['bytes'] for f in files),'files':files},ensure_ascii=False,indent=2))
print('Original archive hash and all 47 original file sizes/CRCs verified; SHA256 captured. No data processing.',flush=True)
