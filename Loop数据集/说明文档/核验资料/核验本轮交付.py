"""Verify documentation coverage and migration metadata; do not process patient rows."""
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,re,stat
P=Path(__file__).resolve().parent; BASE=P.parents[1]; ROOT=BASE.parent
raw=BASE/'原始数据'; docs=BASE/'说明文档'; issues=[]
m=json.loads((P/'迁移清单.json').read_text())
for old,new in m['moves']:
 assert not (ROOT/old).exists(),old
 assert (ROOT/new).exists(),new
for f in m['files']:
 p=ROOT/f['new']; s=p.stat()
 assert (s.st_size,s.st_ino,s.st_mtime_ns)==(f['size'],f['inode'],f['mtime_ns']),str(p)
 if f.get('sha256_before'):assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256_before'],str(p)
manifest=json.loads((P/'原始文件校验.json').read_text())
assert len(manifest['files'])==47 and all(f['matches_official_zip_size_crc'] for f in manifest['files'])
overview=(docs/'01_数据集与全部文件说明.md').read_text()
official_links=re.findall(r'\]\(<\.\./原始数据/([^>]+)>\)',overview)
assert sorted(official_links)==sorted(f['file'] for f in manifest['files'])
schema=json.loads((P/'实际字段清单.json').read_text())
core=(docs/'02_数据表字段字典.md').read_text(); survey=(docs/'03_Surveys全部330字段说明.md').read_text()
survey_fields=re.findall(r'^\| (\d+) \| ([^|]+) \|',survey,re.M)
assert len(survey_fields)==330
assert [int(n) for n,k in survey_fields]==list(range(1,331))
logical={}
for t in schema['tables']:
 with (raw/'Data Tables'/t['file']).open('rb') as h:head=h.readline().decode('utf-8-sig').rstrip('\r\n').split('|')
 assert head==[f['field'] for f in t['fields']],t['file']
 assert len(head)==t['columns']
 if t['logical_table']=='Surveys':assert head==[k.strip() for n,k in survey_fields]
 else:
  section=core.split('## '+t['logical_table']+'：',1)[1].split('\n## ',1)[0]
  actual=re.findall(r'^\| ([^|]+) \|',section,re.M)[1:]
  assert head==[k.strip() for k in actual],t['file']
 logical.setdefault(t['logical_table'],head)
 assert logical[t['logical_table']]==head
assert len(schema['tables'])==25 and len(logical)==18
assert sum(map(len,logical.values()))==596
assert schema['dictionary_only_fields']==[['Surveys.txt',['typical_cgm_locationf']]]
assert {k for name,k in schema['dictionary_missing_fields']}=={'PtID'}
links=0
for p in [BASE/'00_从这里开始.md',BASE/'历史资料/00_历史资料说明.md',*docs.glob('*.md')]:
 s=p.read_text()
 for link in re.findall(r'\]\(([^\n]+?)\)',s):
  link=link.strip('<>')
  if link.startswith('#'):
   if re.fullmatch(r'#c\d+',link):assert '### '+link[1:].upper() in s,link
   continue
  if link.startswith(('http:','https:')):continue
  dest=unquote(link.split('#')[0]);links+=1
  assert (p.parent/dest).exists(),(str(p),dest)
assert stat.S_IMODE((BASE/'历史资料/私密申请记录').stat().st_mode)==0o700
ignore=(ROOT/'.gitignore').read_text()
assert all(x in ignore for x in ['Loop数据集/原始数据/','Loop数据集/官方压缩包/','Loop数据集/历史资料/'])
log=(ROOT/'工作记录.md').read_text();marker='## 已完成 Harness 功能清单'
assert log.count(marker)==1
assert hashlib.sha256(log[log.index(marker):].encode()).hexdigest()==json.loads((P/'工作记录末尾校验.json').read_text())['sha256']
result={'date':'2026-09-14','moved_files_preserved':len(m['files']),'official_files_catalogued':len(official_links),'official_files_bytes':manifest['files_bytes'],'physical_txt_files':25,'logical_tables':18,'logical_field_occurrences_documented':596,'physical_field_occurrences_documented':sum(t['columns'] for t in schema['tables']),'survey_fields_documented':330,'local_links_checked':links,'known_dictionary_only_field':'typical_cgm_locationf','prior_full_crc_sha_evidence':'原始文件校验.json','private_directory_mode':'0700','harness_inventory_unchanged':True,'scope':'Headers, documents, migration metadata and integrity evidence only. No patient-row processing or model training.','issues':issues}
(P/'本轮交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
