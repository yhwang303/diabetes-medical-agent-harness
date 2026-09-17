"""Structural/link/source QA. Does not claim browser visual rendering."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,ast,csv
ROOT=Path(__file__).resolve().parent
class Check(HTMLParser):
 def __init__(self):super().__init__();self.stack=[];self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if 'id' in attrs:self.ids.append(attrs['id'])
  if tag in ['a','img']:self.links.append(attrs.get('href',attrs.get('src','')))
  if tag in ['details','table','tr','td','th','section','body']:self.stack.append(tag)
 def handle_endtag(self,tag):
  if tag in ['details','table','tr','td','th','section','body']:
   assert self.stack and self.stack[-1]==tag,(tag,self.stack[-5:]);self.stack.pop()
p=Check();p.feed((ROOT/'训练审核报告.html').read_text());assert not p.stack;assert len(p.ids)==len(set(p.ids));checked=0
for link in p.links:
 parsed=urlsplit(link)
 if parsed.scheme:continue
 if parsed.path:assert (ROOT/unquote(parsed.path)).resolve().exists(),link;checked+=1
 elif parsed.fragment:assert unquote(parsed.fragment) in p.ids,link
stages=[]
for d in sorted((ROOT/'results').glob('[DT]*')):
 if not (d/'completion.json').exists() or not (d/'summary.json').exists():continue
 rows=json.loads((d/'summary.json').read_text());c=json.loads((d/'completion.json').read_text());assert c['jobs']==len(rows);keys={(r['method'],r['patient'],r['seed'],r['factor']) for r in rows};assert len(keys)==len(rows);assert all((ROOT/r['file']).exists() for r in rows);stages.append({'stage':d.name,'trajectories':len(rows)})
for path in ROOT.glob('*.py'):ast.parse(path.read_text())
for stem,stage in [('完整开发','D23_full_development_comparison'),('封存主比较','T01_frozen_sealed_comparison')]:
 pth=ROOT/('论文对比表_'+stem+'.csv')
 if not pth.exists():continue
 rows=list(csv.DictReader(pth.open()));raw=json.loads((ROOT/'results'/stage/'summary.json').read_text());assert len(rows)==7 and sum(int(r['trajectories']) for r in rows)==len(raw);assert all(r['training_seed']=='260915' and r['training_updates']=='50000' for r in rows if r['method']!='nominal')
result={'html_structure_balanced':True,'resolved_local_links':checked,'completed_stages':stages,'total_archived_trajectory_runs_including_repeats':sum(x['trajectories'] for x in stages),'python_AST_valid':True,'main_tables_single_seed_50k_verified':True,'full_browser_visual_inspection':False,'note':'Browser file URL security policy blocked visual inspection; not bypassed. Scientific images inspected separately.'};(ROOT/'results/report_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='completed_stages'},indent=2))
