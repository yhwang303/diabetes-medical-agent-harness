import json
from pathlib import Path
R=Path(__file__).resolve().parent
rows={}
for p in sorted((R/'results').glob('C_*')):
 n=len(list(p.glob('*_p??_s*.json')));done=(p/'summary.json').exists();rows[p.name]={'episodes':n,'completed':done}
print(json.dumps({'episodes':sum(v['episodes'] for v in rows.values()),'planned':1200,'panels':rows},ensure_ascii=False))
