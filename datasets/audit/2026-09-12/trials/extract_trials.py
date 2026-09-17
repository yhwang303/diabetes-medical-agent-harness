import pathlib,zipfile,hashlib,json,stat
ROOT=pathlib.Path.cwd(); OUT=ROOT/'datasets/audit/2026-09-12/trials'
for ds in ('dclp3','iobp2'):
 src=next((ROOT/'datasets/raw'/ds).glob('*.zip')); dst=ROOT/'datasets/extracted'/ds
 h=hashlib.file_digest(src.open('rb'),'sha256').hexdigest(); files=[]
 with zipfile.ZipFile(src) as z:
  for i in z.infolist():
   p=pathlib.PurePosixPath(i.filename)
   if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16): raise ValueError(i.filename)
   target=dst.joinpath(*p.parts)
   if i.is_dir(): target.mkdir(parents=True,exist_ok=True);continue
   target.parent.mkdir(parents=True,exist_ok=True)
   sha=hashlib.sha256(); n=0
   with z.open(i) as r,target.open('wb') as w:
    while chunk:=r.read(1024*1024): w.write(chunk);sha.update(chunk);n+=len(chunk)
   assert n==i.file_size
   files.append({'path':i.filename,'bytes':n,'crc32':f'{i.CRC:08x}','sha256':sha.hexdigest()})
 (OUT/f'{ds}_extraction.json').write_text(json.dumps({'source':str(src.relative_to(ROOT)),'archive_sha256':h,'safe_paths':True,'all_crc_and_size_valid':True,'files':files,'files_count':len(files),'uncompressed_bytes':sum(x['bytes'] for x in files)},indent=2))
 print(ds,len(files),'complete',flush=True)
