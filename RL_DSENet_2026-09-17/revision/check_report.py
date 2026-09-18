"""Static delivery checks; does not claim browser rendering or TeX compilation."""
from html.parser import HTMLParser
from pathlib import Path
import csv,json,re,hashlib
from urllib.parse import urlparse,unquote
ROOT=Path(__file__).resolve().parent
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[];self.tables=[];self.current=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  for key in ('href','src'):
   if key in a:self.links.append(a[key])
  if tag=='table':self.current=0
  if tag=='tr' and self.current is not None:self.current+=1
 def handle_endtag(self,tag):
  if tag=='table':self.tables.append(self.current);self.current=None
p=Parser();doc=(ROOT/'论文正文对比报告.html').read_text();p.feed(doc)
missing=[]
for link in p.links:
 u=urlparse(link)
 if u.scheme:continue
 if u.path:
  if not (ROOT/unquote(u.path)).exists():missing.append(link)
 elif u.fragment and u.fragment not in p.ids:missing.append(link)
assert not missing,missing
assert len(p.ids)==len(set(p.ids))
for panel in ('main','native'):
 rows=list(csv.reader((ROOT/'paper_tables'/f'{panel}_comparison.csv').open()))
 assert len(rows)==11 and all(len(r)==15 for r in rows)
 assert [r[0] for r in rows[1:]]==['Constant basal','BC','TD3+BC','IQL','ReBRAC','FQL','LOM','GFP','RL-DITR*','DSENet-RL (ours)']
 for suffix,ncol in (('',15),('_compact',11)):
  tex=(ROOT/'paper_tables'/f'{panel}_comparison{suffix}.tex').read_text()
  for env in ('table*','tabular'):assert len(re.findall(re.escape('\\begin{'+env+'}'),tex))==len(re.findall(re.escape('\\end{'+env+'}'),tex))==1
  lines=[l for l in tex.splitlines() if l.endswith('\\\\')]
  assert len(lines)==11 and all(l.count(' & ')==ncol-1 for l in lines),(panel,suffix)
  assert tex.count('DSENet-RL')==1
for s in ('97.08','89.78','7.30','1.55','0.66','12.39','5.66','2026','2025','L+S','严格同训练数据'):assert s in doc,s
assert p.tables.count(11)>=2,p.tables
r={'local_links_checked':sum(not urlparse(l).scheme for l in p.links),'missing_links':missing,'main_and_native_rows_each':10,'full_table_columns':15,'compact_latex_columns':11,'latex_structure_passed':True,'html_static_checks_passed':True,'browser_visual_check':False,'latex_compiled':False,'figures_visually_inspected':True,'html_sha256':hashlib.sha256(doc.encode()).hexdigest()}
(ROOT/'checks/report_static_qa.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
