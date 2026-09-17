"""Independent official physiological simulator; policies see only reconstructed observable history."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import argparse,json,datetime,hashlib,time
import numpy as np
import pandas as pd
import pkg_resources
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Action
from numpy_policy import NumpyPolicy
ROOT=Path(__file__).resolve().parent
NORMALIZER=ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json'

from observable_history import History

def score(bg):
 b=np.asarray(bg);risk=10*(1.509*(np.log(np.maximum(b,1))**1.084-5.381))**2
 return np.where(b<70,-1,1-np.clip(risk,0,15.5)/7.75),risk

def simulate(patient_id,seed,policy_path=None,days=3,bolus_factor=1.,mode='nominal'):
 rng=np.random.default_rng(seed);start=datetime.datetime(2020,1,1);meals=[];announcements={}
 # Fixed paired event generation depends only on scenario seed, never on policy.
 for day in range(days):
  for hour,grams in [(8,45),(13,65),(19,70)]:
   minute=day*1440+hour*60+int(rng.integers(-6,7))*5;amount=float(np.clip(grams+rng.normal(0,10),20,120))
   meals.append((minute/60,amount));announcements[minute]=amount
 name='adult#%03d'%patient_id;sensor=CGMSensor.withName('GuardianRT',seed=seed+10000);pump=InsulinPump.withName('Insulet')
 patient=T1DPatient.withName(name);env=T1DSimEnv(patient,sensor,pump,CustomScenario(start_time=start,scenario=meals));first=env.reset();assert env.sample_time==5
 params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'));row=params.loc[params.Name==name].iloc[0]
 quest=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'));cr=float(quest.loc[quest.Name==name,'CR'].iloc[0]);nominal=float(row.u2ss*row.BW/6000*60)
 policy=NumpyPolicy(policy_path) if policy_path else None;history=History();history.append(0,float(first.observation.CGM));records=[];failed=False;previous_time=env.time
 noise_rng=np.random.default_rng(seed+20000);begin=time.time()
 for step in range(days*288):
  minute=step*5;amount=announcements.get(minute,0.);bolus_u=bolus_factor*amount/cr
  if minute<360 or mode=='nominal':requested=nominal
  elif policy is not None:requested=float(10*(policy(history.state(),noise_rng.normal(size=(1,1))) [0,0]+1))
  else:raise ValueError('Unknown policy')
  if not np.isfinite(requested) or not 0<=requested<=20.00001:raise ValueError('Nonfinite or out-of-contract policy action')
  delivered_basal=float(pump.basal(requested/60));delivered_bolus=float(pump.bolus(bolus_u/5))
  transition=env.step(Action(basal=requested/60,bolus=bolus_u/5));elapsed=(env.time-previous_time).total_seconds()/60;assert elapsed==5;previous_time=env.time
  # Independent evaluator may log true BG. It is never passed to History or NumpyPolicy.
  cgm=float(transition.observation.CGM);bg=float(transition.info['bg']);food=float(transition.info['meal'])*5
  delivered_total=float(env.insulin_hist[-1])*5;assert abs(delivered_total-(delivered_basal+delivered_bolus)*5)<1e-8
  history.append(minute+5,cgm,delivered_basal*5,delivered_bolus*5 if delivered_bolus>0 else None,food if food>0 else None)
  records.append({'minute':minute+5,'cgm_mg_dl':cgm,'bg_mg_dl':bg,'requested_basal_u_h':requested,'delivered_basal_u_h':delivered_basal*60,'bolus_u':delivered_bolus*5,'meal_g':food,'warmup':minute<360})
  if transition.done or not np.isfinite(bg):failed=True;break
 measured=[r for r in records if not r['warmup']];bg=np.array([r['bg_mg_dl'] for r in measured]);cgm=np.array([r['cgm_mg_dl'] for r in measured]);planned=days*288-72
 # Conservative absorbing failure accounting preserves the fixed denominator; never silently drops failed time.
 missing=planned-len(bg)
 if missing:
  fill=600. if not len(bg) or not np.isfinite(bg[-1]) or bg[-1]>600 else 10.
  bg=np.concatenate([bg,np.full(missing,fill)]);cgm=np.concatenate([cgm,np.full(missing,fill)])
 status,risk=score(bg)
 metrics={'tir_pct':float(((bg>=70)&(bg<=180)).mean()*100),'tbr70_pct':float((bg<70).mean()*100),'tbr54_pct':float((bg<54).mean()*100),'tar180_pct':float((bg>180).mean()*100),'tar250_pct':float((bg>250).mean()*100),'mean_bg_mg_dl':float(bg.mean()),'bg_sd_mg_dl':float(bg.std()),'mean_risk':float(risk.mean()),'status_score_mean':float(status.mean()),'cgm_tir_pct':float(((cgm>=70)&(cgm<=180)).mean()*100),'basal_u_per_day':sum(r['delivered_basal_u_h']/12 for r in measured)/(planned/288),'bolus_u_per_day':sum(r['bolus_u'] for r in measured)/(planned/288),'failed':failed,'missing_failure_steps':missing,'measured_steps':len(measured),'fixed_metric_denominator':planned,'pump_rate_changed_fraction':float(np.mean([abs(r['requested_basal_u_h']-r['delivered_basal_u_h'])>1e-7 for r in measured])) if measured else 0}
 return {'patient':name,'scenario_seed':seed,'policy':str(policy_path) if policy_path else 'nominal_basal','days':days,'bolus_factor':bolus_factor,'nominal_basal_u_h':nominal,'metrics':metrics,'wall_seconds':time.time()-begin,'records':records,'future_information_to_policy':False,'exercise_physiology_validated':False,'clinical_ready':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--policy');ap.add_argument('--patient',type=int,default=1);ap.add_argument('--seed',type=int,default=3101);ap.add_argument('--days',type=int,default=3);ap.add_argument('--bolus-factor',type=float,default=1);ap.add_argument('--output',required=True);args=ap.parse_args()
 if args.patient>4:raise RuntimeError('Sealed adults require a separately frozen final-evaluation entrypoint')
 result=simulate(args.patient,args.seed,args.policy,args.days,args.bolus_factor,'learned' if args.policy else 'nominal');out=ROOT/args.output;out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
