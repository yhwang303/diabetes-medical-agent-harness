import json
from pathlib import Path
from aggregate import aggregate
R=Path(__file__).resolve().parent
for name in ['C_native_hold','C_native_ours','C_brake17_hold','C_brake17_ours']:
 p=R/'results'/name
 if (p/'summary.json').exists():
  a=aggregate(p);print(json.dumps({'name':name,'failures':a['failures'],**{k:a[k] for k in ['tir_lower_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','sd_mg_dl','cv_pct']}}))
