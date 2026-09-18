import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def function_source(p,name):
    s=p.read_text();tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.dump(node,include_attributes=False)
for name in ('scenario','environment_worker'):
    assert function_source(ROOT/'evaluate_extended.py',name)==function_source(ROOT.parent/'evaluate_control.py',name)
new=json.loads((ROOT/'results/check_hold_equivalence/normal_p01_s91701.json').read_text())
old=json.loads((ROOT.parent/'results/F01_hold/normal_p01_s91701.json').read_text())
assert new['records']==old['records'] and new['metrics']==old['metrics']
scorer=ROOT.parents[1]/'RL_DITR创新_2026-09-16/control_metrics.py'
freeze=json.loads((ROOT.parent/'configs/final_freeze.json').read_text())
digest=hashlib.sha256(scorer.read_bytes()).hexdigest()
assert digest==freeze['source_sha256'][str(scorer.relative_to(ROOT.parents[1]))]
result={'status':'passed','simulator_function_ast_unchanged':True,'scenario_function_ast_unchanged':True,
        'hold_3day_raw_records_exact':True,'hold_metrics_exact':True,'scorer_sha256':digest}
(ROOT/'checks/evaluator_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
