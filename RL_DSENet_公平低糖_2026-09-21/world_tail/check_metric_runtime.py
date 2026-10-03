"""Diagnose exact scorer differences without changing the scorer or assertion."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
P = R.parent.parent
spec = importlib.util.spec_from_file_location('unchanged_metrics', P/'RL_DITR创新_2026-09-16/control_metrics.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
directory = P/'RL_DSENet_2026-09-17/results/F11_actor'
s = json.loads((directory/'summary.json').read_text()); diffs=[]

def compare(a, b, prefix=''):
    if isinstance(a, dict):
        for k in set(a)|set(b):
            if k not in a or k not in b: diffs.append(dict(path=prefix+k, missing=True)); continue
            compare(a[k], b[k], prefix+k+'.')
    elif a != b:
        diffs.append(dict(path=prefix, recomputed=a, saved=b,
                          absolute_difference=abs(a-b) if isinstance(a, (float, int)) and isinstance(b, (float,int)) else None))

for e in s['episodes']:
    d=json.loads((directory/(e['key']+'.json')).read_text())
    actual=module.summarize(d['records'],(d['job']['total_minutes']-360)//5,d['failure_reason'] is not None)
    compare(actual,d['metrics'],e['key']+'.')
out=dict(python=sys.version,numpy=np.__version__,differences=diffs,count=len(diffs))
(R/'checks'/('scorer_runtime_'+str(sys.version_info[1])+'.json')).write_text(json.dumps(out,indent=2))
print(json.dumps(dict(python=sys.version,numpy=np.__version__,count=len(diffs),first=diffs[:5])),flush=True)
