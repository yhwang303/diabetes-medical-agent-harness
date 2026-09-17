"""Development-only common-state, common-disturbance 4h intervention benchmark."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
import argparse,copy,datetime,hashlib,json,sys
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RL进阶对比_2026-09-15'))
from observable_history import History
from evaluate_control import scenario

def patient_probe(pid,dynamic=False):
 import pandas as pd
 import pkg_resources
 from simglucose.patient.t1dpatient import T1DPatient
 from simglucose.sensor.cgm import CGMSensor
 from simglucose.actuator.pump import InsulinPump
 from simglucose.simulation.scenario import CustomScenario
 from simglucose.simulation.env import T1DSimEnv
 from simglucose.controller.base import Action
 seed=3101;meals=scenario(seed,4320);announcements=dict(meals);name='adult#%03d'%pid
 params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'));row=params.loc[params.Name==name].iloc[0];nominal=float(row.u2ss*row.BW/6000*60)
 quest=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'));cr=float(quest.loc[quest.Name==name,'CR'].iloc[0])
 env=T1DSimEnv(T1DPatient.withName(name),CGMSensor.withName('GuardianRT',seed=seed+10000),InsulinPump.withName('Insulet'),CustomScenario(datetime.datetime(2020,1,1),[(m/60,g) for m,g in meals]));first=env.reset();history=History();history.append(0,float(first.observation.CGM));samples=[]
 def advance(environment,minute,rate,hist=None):
  bolus=announcements.get(minute,0)/cr;db=float(environment.pump.basal(rate/60));du=float(environment.pump.bolus(bolus/5));tr=environment.step(Action(basal=rate/60,bolus=bolus/5));food=float(tr.info['meal'])*5
  if hist is not None:hist.append(minute+5,float(tr.observation.CGM),db*5,du*5 if du>0 else None,food if food>0 else None)
  return {'bg':float(tr.info['bg']),'cgm':float(tr.observation.CGM),'delivered_basal_u_h':db*60,'bolus_u':du*5,'meal_g':food,'done':bool(tr.done)}
 for minute in range(0,4320,5):
  if minute in [360,720,1080,1440,2160,2880,3600]:
   arms=[]
   plans=[('constant_%s'%delta,[delta]*48) for delta in [0.,0.,-.5,-.25,.25,.5]]
   if dynamic:plans=[('reference',[0.]*48),('reference',[0.]*48),('early_plus',[.5]*24+[0.]*24),('late_plus',[0.]*24+[.5]*24),('early_minus',[-.5]*24+[0.]*24),('late_minus',[0.]*24+[-.5]*24),('plus_then_minus',[.5]*24+[-.5]*24),('minus_then_plus',[-.5]*24+[.5]*24)]
   for pattern,deltas in plans:
    branch=copy.deepcopy(env);rates=[max(0,nominal+delta) for delta in deltas];records=[]
    for k in range(48):
     record=advance(branch,minute+k*5,rates[k]);records.append(record)
     if record['done']:break
    arm={'requested_actions_u_h':rates,'records':records}
    if dynamic:arm['pattern']=pattern
    else:arm['delta_u_h']=deltas[0]
    arms.append(arm)
   assert arms[0]==arms[1],'Identical cloned interventions diverge'
   for arm in arms[2:]:
    for reference,item in zip(arms[0]['records'],arm['records']):assert reference['meal_g']==item['meal_g'] and reference['bolus_u']==item['bolus_u']
   samples.append({'patient':pid,'minute':minute,'history':history.state().tolist(),'arms':[arms[0]]+arms[2:],'true_zero_duplicate_exact':True,'common_exogenous_events_verified':True})
  if advance(env,minute,nominal,history)['done']:raise RuntimeError('Reference environment terminated')
 return samples

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dynamic',action='store_true');args=ap.parse_args()
 out=ROOT/('action_probe_dynamic' if args.dynamic else 'action_probe');out.mkdir(exist_ok=False)
 with ProcessPoolExecutor(max_workers=4) as pool:rows=sum(list(pool.map(partial(patient_probe,dynamic=args.dynamic),[1,2,3,4])),[])
 p=out/'samples.json';p.write_text(json.dumps(rows,allow_nan=False));manifest={'scope':'exposed development only; evaluator hidden states never provided to learned models','patients':[1,2,3,4],'seed':3101,'histories':len(rows),'horizon_minutes':240,'deltas_u_h':[0,-.5,-.25,.25,.5],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'samples_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'training_use':False,'shortened_failed_branches_retained':True}
 if args.dynamic:
  manifest.pop('deltas_u_h');manifest['patterns']=[a['pattern'] for a in rows[0]['arms']];manifest['purpose']='development diagnostic for delayed response to reordered equal-dose action plans; never training labels'
  checks=[]
  for row in rows:
   arms={a['pattern']:a for a in row['arms']}
   for first,second in [('early_plus','late_plus'),('early_minus','late_minus'),('plus_then_minus','minus_then_plus')]:
    a=arms[first];b=arms[second]
    if len(a['records'])==48 and len(b['records'])==48:
     da=sum(x['delivered_basal_u_h']/12 for x in a['records']);db=sum(x['delivered_basal_u_h']/12 for x in b['records']);assert abs(da-db)<1e-10
     checks.append({'patient':row['patient'],'minute':row['minute'],'patterns':[first,second],'delivered_total_u_difference':da-db})
  manifest['equal_total_dose_complete_pairs']=len(checks);(out/'equal_dose_checks.json').write_text(json.dumps(checks,indent=2))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
