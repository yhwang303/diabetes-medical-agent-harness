"""Check audit deliverables and extracted manifest metadata, without processing patient rows."""
from pathlib import Path
from collections import Counter
from urllib.parse import unquote
import json,re,zipfile
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
issues=[]; sets={}
loop=json.loads((root/'Loop数据集/历史资料/2026-09-12审计/extraction_manifest.json').read_text())
sets['loop']=[(x['path'],x['bytes']) for x in loop]
assert all(x['validated'] for x in loop)
for ds in ['dclp3','iobp2']:
 m=json.loads((out/f'trials/{ds}_extraction.json').read_text())
 assert m['safe_paths'] and m['all_crc_and_size_valid']
 sets[ds]=[(x['path'],x['bytes']) for x in m['files']]
for ds in ['ohio','uom']:
 m=json.loads((out/'ohio_uom/extraction_manifest.json').read_text())
 sets[ds]=[(x['member'],x['bytes']) for x in m if x['dataset']==ds]
for ds,files in sets.items():
 for name,size in files:
  p=(root/'Loop数据集/原始数据'/name) if ds=='loop' else (root/'datasets/extracted'/ds/name)
  if not p.is_file() or p.stat().st_size!=size:issues.append(f'extracted mismatch:{ds}/{name}')
counts={ds:len(x) for ds,x in sets.items()}
assert counts=={'loop':47,'dclp3':87,'iobp2':227,'ohio':24,'uom':127},counts
assert sum(counts.values())==512
json_count=0
for p in out.rglob('*.json'):
 if p.name=='delivery_verification.json':continue
 json.loads(p.read_text());json_count+=1
compiled=[]
for p in out.rglob('*.py'):
 compile(p.read_text(),str(p),'exec');compiled.append(str(p.relative_to(out)))
linked_docs=['07_真实数据集RL训练资格审计报告.md','06_RL真实联合数据集下载与构建方案.md','datasets/README.md']
linked_docs += [str(p.relative_to(root)) for p in out.rglob('*.md') if 'sources' not in p.relative_to(out).parts]
checked=0
for name in linked_docs:
 p=root/name;s=p.read_text()
 for target in re.findall(r'\]\(([^\n]+?)\)',s):
  target=target.strip('<>')
  if target.startswith(('http:','https:','#','app:','mailto:')):continue
  dest=unquote(target.split('#')[0])
  if not dest:continue
  checked+=1
  if not (p.parent/dest).exists():issues.append(f'link:{name}:{dest}')
log=(root/'工作记录.md').read_text()
assert log.count('## 已完成 Harness 功能清单')==1
result={'scope':'Metadata/links/syntax only. Does not rerun data quality processing, models or engineering tests.','extracted_file_counts':counts,'extracted_files_size_checked':sum(counts.values()),'json_files_readable':json_count,'python_files_syntax_checked':len(compiled),'local_links_checked':checked,'issues':issues}
(out/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
if issues:raise SystemExit(1)
