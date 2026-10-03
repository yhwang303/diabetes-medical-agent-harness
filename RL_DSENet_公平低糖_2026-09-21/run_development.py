import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;B=P/'RL_DSENet_2026-09-17';py=str(P/'.venv/bin/python')
def run(name,mode,config='development',brake=None):
 cmd=[py,str(R/'evaluate.py'),'--name',name,'--config',str(R/'configs'/(config+'.json')),'--mode',mode,'--batch-size','20']
 if mode=='actor':cmd+=['--bounded-reference','--checkpoint',str(B/'results/D06_selected_policy/policy.pt')]
 if brake:cmd+=['--brake-config',str(R/'configs'/(brake+'.json'))]
 subprocess.run(cmd,check=True)
subprocess.run([py,str(R/'check_mechanics.py')],check=True)
freeze={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(R.glob('*.py'))+list((R/'configs').glob('*.json'))+list(R.glob('*.md'))}
(R/'configs/development_freeze.json').write_text(json.dumps(freeze,indent=2))
for mode,folder in [('hold','F01_hold'),('actor','F11_actor')]:
 run('check_'+mode,mode,'smoke')
 a=json.loads((R/'results'/('check_'+mode)/'normal_p01_s91701.json').read_text());b=json.loads((B/'results'/folder/'normal_p01_s91701.json').read_text())
 assert a['records']==b['records'] and a['metrics']==b['metrics'],mode
(R/'checks/equivalence.json').write_text(json.dumps({'status':'passed','hold_actor_3day_records_metrics_exact':True}))
for name in ('brake17','brake30'):run('D_'+name,'actor',brake=name)
print('DEVELOPMENT_COMPLETE',flush=True)
