from html.parser import HTMLParser
from pathlib import Path
import json
P=Path(__file__).resolve().parent
class Tables(HTMLParser):
 def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
 def handle_starttag(self,tag,attrs):
  if tag=='table':self.table=[]
  elif tag=='tr':self.row=[]
  elif tag in ('td','th'):self.cell={'text':'','attrs':dict(attrs)}
  elif tag in ('p','br') and self.cell is not None:self.cell['text']+='\n'
 def handle_data(self,data):
  if self.cell is not None:self.cell['text']+=data
 def handle_endtag(self,tag):
  if tag in ('td','th') and self.cell is not None:self.cell['text']=self.cell['text'].strip();self.row.append(self.cell);self.cell=None
  elif tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
  elif tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None
p=Tables();p.feed((P/'官方字典转换.html').read_text());(P/'官方字典表格.json').write_text(json.dumps(p.tables,ensure_ascii=False,indent=2))
for i,t in enumerate(p.tables):print(i,len(t),[[c['text'] for c in r] for r in t[:2]])
# Expand HTML rowspans explicitly so inherited answer codes are not lost.
expanded=[]
for tab in p.tables:
 grid={}
 for ri,row in enumerate(tab):
  ci=0
  for cell in row:
   while (ri,ci) in grid:ci+=1
   nr=int(cell['attrs'].get('rowspan',1));nc=int(cell['attrs'].get('colspan',1))
   for dr in range(nr):
    for dc in range(nc):grid[(ri+dr,ci+dc)]=cell['text']
   ci+=nc
 expanded.append([[grid.get((r,c),'') for c in range(max([c for rr,c in grid if rr==r],default=-1)+1)] for r in range(len(tab))])
(P/'官方字典展开.json').write_text(json.dumps(expanded,ensure_ascii=False,indent=2))
with (P/'问卷字段供核对.txt').open('w') as f:
 for row in expanded[15][1:]:
  if len(row)>1 and row[0]!=row[1]:f.write(row[0]+' : '+row[1].replace('\n',' ')+' | '+(row[4].replace('\n','; ') if len(row)>4 else '')+'\n')
