"""Independent counterfactual development probes. Only the evaluator clones simulator state."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import copy,json,datetime,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import pkg_resources
from sim_eval import T1DPatient,CGMSensor,InsulinPump,CustomScenario,T1DSimEnv,Action,ROOT
from observable_history import History

def one_patient(pid):
 seed=3101;rng=np.random.default_rng(seed);start=datetime.datetime(2020,1,1);meals=[];announcements={}
 for day in range(3):
  for hour,grams in [(8,45),(13,65),(19,70)]:
   minute=day*1440+hour*60+int(rng.integers(-6,7))*5;amount=float(np.clip(grams+rng.normal(0,10),20,120));meals.append((minute/60,amount));announcements[minute]=amount
 name='adult#%03d'%pid;p=T1DPatient.withName(name);env=T1DSimEnv(p,CGMSensor.withName('GuardianRT',seed=seed+10000),InsulinPump.withName('Insulet'),CustomScenario(start,meals));first=env.reset();history=History();history.append(0,float(first.observation.CGM));params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'));row=params.loc[params.Name==name].iloc[0];nominal=float(row.u2ss*row.BW/6000*60);quest=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'));cr=float(quest.loc[quest.Name==name,'CR'].iloc[0]);samples=[]
 def advance(e,minute,rate,h=None):
  bu=announcements.get(minute,0.)/cr;db=float(e.pump.basal(rate/60));du=float(e.pump.bolus(bu/5))*5;t=e.step(Action(basal=rate/60,bolus=bu/5));assert abs(float(e.insulin_hist[-1])*5-(db*5+du))<1e-8
  if h is not None:h.append(minute+5,float(t.observation.CGM),db*5,du if du>0 else None,float(t.info['meal'])*5 if t.info['meal']>0 else None)
  return t
 for step in range(3*288):
  minute=step*5
  if minute in [360,720,1080,1440,2160,2880,3600]:
   # Deep copies include integrator, digestion and sensor RNG state; zero-intervention branches must agree exactly.
   branches=[];event_rows=[]
   for delta in [0.,0.,.1,1.]:
    branch=copy.deepcopy(env);bh=copy.deepcopy(history);bg=[];cgm=[];delivered=[]
    for j in range(12):
     t=advance(branch,minute+j*5,nominal+delta,bh);bg.append(float(t.info['bg'])/18);cgm.append(float(t.observation.CGM)/18);delivered.append(float(branch.pump.basal((nominal+delta)/60))*60)
    branches.append({'delta_requested_u_h':delta,'bg_mmol_l':bg,'cgm_mmol_l':cgm,'actual_basal_u_h':delivered});event_rows.append(np.array(bh.rows[-12:])[:,[2,7,3,8,4,9]])
   assert branches[0]==branches[1],'Cloned zero-intervention trajectories diverged'
   assert all(np.array_equal(event_rows[0],e) for e in event_rows),'Common recorded interventions changed across basal arms'
   samples.append({'patient':name,'minute':minute,'state':history.state().tolist(),'events':event_rows[0].tolist(),'arms':[branches[0],branches[2],branches[3]],'no_hidden_state_to_model':True})
  t=advance(env,minute,nominal,history)
  if t.done:raise RuntimeError('Unexpected nominal reference failure')
 return samples

def main():
 dest=ROOT/'results/independent_dose_probe';dest.mkdir(exist_ok=True,parents=True)
 if (dest/'samples.json').exists():raise RuntimeError('Preserve previous probe')
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(one_patient,[1,2,3,4]))
 rows=sum(results,[]);p=dest/'samples.json';p.write_text(json.dumps(rows));meta={'patients':[1,2,3,4],'scenario_seed':3101,'histories':len(rows),'deltas_u_h':[.1,1.],'horizon_minutes':60,'common_bolus_factor':1.,'zero_intervention_clone_exact':True,'joint_events_identical_across_arms':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Development-only independent physiological counterfactual probes; model receives observable history and paired common-event context, no simulator state','clinical_ready':False};(dest/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
