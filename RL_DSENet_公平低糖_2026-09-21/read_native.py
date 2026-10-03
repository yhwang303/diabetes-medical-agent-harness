import json
from pathlib import Path
from aggregate import aggregate
R=Path(__file__).resolve().parent
out={}
for m in json.loads((R/'configs/final_manifest.json').read_text())['methods']:
 a=aggregate(R/'results'/('C_native_'+m['key']));out[m['key']]={'failures':a['failures'],**{k:a[k] for k in ['tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','sd_mg_dl','cv_pct']}}
(R/'analysis/native_ready.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
