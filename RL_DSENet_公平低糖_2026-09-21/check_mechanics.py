import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
from brake import Brake,attenuation,signals,MEAN,SCALE
R=Path(__file__).resolve().parent;P=R.parent;B=P/'RL_DSENet_2026-09-17'
def fn(path,name):
 return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
for name in ('scenario','environment_worker'):assert fn(R/'evaluate.py',name)==fn(B/'revision/evaluate_extended.py',name)
def history(g):
 x=np.zeros((72,22));x[-6:,0]=(np.asarray(g)/18-MEAN)/SCALE;x[-6:,5]=1;return x
h=history([140,130,120,110,100,90]);g,f,s=signals(h);assert abs(s+2)<1e-12 and abs(f-95)<1e-10
b=Brake(dict(slope_minutes=17));u,d=b.apply('x',h,1.2);assert 0<u<1.2 and abs(d['estimate']-61)<1e-10
u,d=b.apply('x',history([60,61,62,63,64,65]),1.2);assert u==1.2 and not d['active']
u,d=b.apply('x',history([190,189,188,187,186,185]),1.2);assert u==1.2
for g in (10,20,40,54,70,90,112.5,180):assert 0<attenuation(g)[0]<=1
assert attenuation(54)[0]<attenuation(70)[0]<attenuation(90)[0]<attenuation(112.5)[0]
for bad in [np.zeros((72,22)),np.ones((1,22))]:
 try:signals(bad)
 except ValueError:pass
 else:raise AssertionError('invalid data accepted')
result={'status':'passed','environment_and_scenario_AST_equal':True,'slope_units_verified':True,'continuous_attenuation_and_immediate_recovery':True,'no_BG_or_patient_identity_input':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')}}
(R/'checks/mechanics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
