import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/'RL_DITR创新_2026-09-16'))
from control_metrics import summarize
METRICS=['tir_lower_bound_pct','tir_upper_bound_pct','tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','sd_mg_dl','cv_pct','lbgi','hbgi','mean_mg_dl','coverage_pct']
def aggregate(folder):
 s=json.loads((folder/'summary.json').read_text());assert s['status']=='completed';patients={};fail=0;long=0;keys=[];hashes={}
 for e in s['episodes']:
  p=folder/(e['key']+'.json');d=json.loads(p.read_text());m=summarize(d['records'],(d['job']['total_minutes']-360)//5,d['failure_reason'] is not None)
  assert m==d['metrics']==e['metrics'];fail+=bool(d['failure_reason']);long+=m['bg'].get('prolonged_under54_120min_events',0);pid=str(d['job']['patient']);patients.setdefault(pid,[]).append(m);keys.append(e['key']);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 means={p:{k:float(np.mean([m['bg'][k] for m in ms if k in m['bg']])) for k in METRICS} for p,ms in patients.items()}
 out={k:dict(mean=float(np.mean([m[k] for m in means.values()])),sd=float(np.std([m[k] for m in means.values()],ddof=1))) for k in METRICS}
 out.update(failures=fail,episodes=len(keys),prolonged54=long,patients=means,keys=sorted(keys),raw_sha256=hashes,folder=str(folder.relative_to(P)))
 return out

def joint(new,old):
 changes={k:new[k]['mean']-old[k]['mean'] for k in METRICS}
 gates={'tir_not_lower':changes['tir_lower_bound_pct']>=-1e-8,'complete':new['failures']==0 and new['coverage_pct']['mean']==100.,'hypo_strict':min(changes[k] for k in ('tbr70_pct','tbr54_pct'))<-1e-8}
 gates.update({k:changes[k]<=1e-8 for k in ['tbr70_pct','tbr54_pct','tar180_pct','tar250_pct','hbgi','sd_mg_dl','cv_pct']})
 return dict(passed=all(gates.values()),gates=gates,changes=changes)
if __name__=='__main__':
 baseline=aggregate(P/'RL_DSENet_2026-09-17/results/F11_actor');out={'old':baseline};gates={}
 for d in sorted((R/'results').glob('D_*')):
  if not (d/'summary.json').exists():continue
  out[d.name]=aggregate(d);gates[d.name]=joint(out[d.name],baseline)
 (R/'analysis/development.json').write_text(json.dumps({'methods':out,'joint':gates},indent=2,allow_nan=False))
 print(json.dumps({k:{m:v[m]['mean'] for m in METRICS[:10]} for k,v in out.items()},indent=2));print('PASS', [k for k,v in gates.items() if v['passed']])
