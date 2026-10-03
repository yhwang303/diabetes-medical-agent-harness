import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
import datetime,json,multiprocessing as mp,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P/'RL进阶对比_2026-09-15'));sys.path.insert(0,str(P/'RL_DITR创新_2026-09-16'))
from observable_history import History
from control_metrics import summarize
from evaluate import scenario

def environment_worker(connection,job):
 import pandas as pd
 import pkg_resources
 from simglucose.patient.t1dpatient import T1DPatient
 from simglucose.sensor.cgm import CGMSensor
 from simglucose.actuator.pump import InsulinPump
 from simglucose.simulation.scenario import CustomScenario
 from simglucose.simulation.env import T1DSimEnv
 from simglucose.controller.base import Action
 records=[];failure=None;begin=time.time();nominal=None
 try:
  name='adult#%03d'%job['patient'];params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'));row=params.loc[params.Name==name].iloc[0]
  quest=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'));cr=float(quest.loc[quest.Name==name,'CR'].iloc[0]);nominal=float(row.u2ss*row.BW/6000*60)
  sensor=CGMSensor.withName('GuardianRT',seed=job['seed']+10000);pump=InsulinPump.withName('Insulet');env=T1DSimEnv(T1DPatient.withName(name),sensor,pump,CustomScenario(datetime.datetime(2020,1,1),[(m/60,g) for m,g in job['meals']]))
  first=env.reset();assert env.sample_time==5;history=History();history.append(0,float(first.observation.CGM));announcements=dict(job['meals']);previous=env.time
  for minute in range(0,job['total_minutes'],5):
   if minute<360:requested=nominal
   else:
    connection.send({'done':False,'history':history.state(),'nominal_u_h':nominal});command=connection.recv()
    if 'abort' in command:raise RuntimeError(command['abort'])
    requested=float(command['action_u_h'])
   if not np.isfinite(requested) or not 0<=requested<=20:raise ValueError('Invalid policy action')
   bolus=job['bolus_factor']*announcements.get(minute,0)/cr;db=float(pump.basal(requested/60));du=float(pump.bolus(bolus/5))
   tr=env.step(Action(basal=requested/60,bolus=bolus/5));assert (env.time-previous).total_seconds()/60==5;previous=env.time
   cgm=float(tr.observation.CGM);bg=float(tr.info['bg']);food=float(tr.info['meal'])*5
   assert abs(float(env.insulin_hist[-1])*5-(db+du)*5)<1e-8
   records.append({'minute':minute+5,'cgm_mg_dl':cgm if np.isfinite(cgm) else None,'bg_mg_dl':bg if np.isfinite(bg) else None,'requested_basal_u_h':requested,'delivered_basal_u_h':db*60,'bolus_u':du*5,'meal_g':food,'warmup':minute<360})
   if not np.isfinite(bg) or not np.isfinite(cgm):failure='nonfinite_environment';break
   history.append(minute+5,cgm,db*5,du*5 if du>0 else None,food if food>0 else None)
   if tr.done:failure='native_environment_done';break
 except Exception as error:failure=repr(error)
 result={'job':job,'nominal_basal_u_h':nominal,'failure_reason':failure,'metrics':summarize(records,(job['total_minutes']-360)//5,failure is not None),'records':records,'wall_seconds':time.time()-begin,'future_information_to_policy':False,'hidden_state_to_policy':False,'exercise_physiology_validated':False}
 connection.send({'done':True,'result':result,'final_history':history.state().tolist()});connection.close()
