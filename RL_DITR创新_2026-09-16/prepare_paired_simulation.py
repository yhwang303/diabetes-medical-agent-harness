"""New training-only simulator factual arms, shared by reference and paired ablations.
No development probe labels or final scenario seeds are consumed.
"""
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

def generate(job,temporal=False):
 import pandas as pd
 import pkg_resources
 from simglucose.patient.t1dpatient import T1DPatient
 from simglucose.sensor.cgm import CGMSensor
 from simglucose.actuator.pump import InsulinPump
 from simglucose.simulation.scenario import CustomScenario
 from simglucose.simulation.env import T1DSimEnv
 from simglucose.controller.base import Action
 pid,seed,bolus_factor,basal_factor=job;name='adult#%03d'%pid;meals=scenario(seed,4320);announcements=dict(meals)
 params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'));row=params.loc[params.Name==name].iloc[0];nominal=float(row.u2ss*row.BW/6000*60)
 quest=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'));cr=float(quest.loc[quest.Name==name,'CR'].iloc[0]);base_rate=nominal*basal_factor
 env=T1DSimEnv(T1DPatient.withName(name),CGMSensor.withName('GuardianRT',seed=seed+10000),InsulinPump.withName('Insulet'),CustomScenario(datetime.datetime(2020,1,1),[(m/60,g) for m,g in meals]));first=env.reset();history=History();history.append(0,float(first.observation.CGM));groups=[];base_failure=False
 def step(environment,hist,minute,rate):
  bolus=bolus_factor*announcements.get(minute,0)/cr;db=float(environment.pump.basal(rate/60));du=float(environment.pump.bolus(bolus/5));tr=environment.step(Action(basal=rate/60,bolus=bolus/5));cgm=float(tr.observation.CGM);bg=float(tr.info['bg']);food=float(tr.info['meal'])*5
  if not np.isfinite([cgm,bg]).all():raise ValueError('Nonfinite simulator outcome; retain generation failure instead of fabricating labels')
  hist.append(minute+5,cgm,db*5,du*5 if du>0 else None,food if food>0 else None)
  assert abs(float(environment.insulin_hist[-1])*5-(db+du)*5)<1e-8
  return cgm/18,bg/18,db*60,du*5,food,bool(tr.done)
 for minute in range(0,4320,5):
  if minute in [360,720,1080,1440,2160,2880,3600]:
   arms=[]
   plans=[np.full(48,delta) for delta in [0.,0.,-.5,-.25,.25,.5]]
   if temporal:
    plans=[np.zeros(48),np.zeros(48)]
    for block in range(3):
     for delta in [-.5,.5]:
      change=np.zeros(48);change[block*16:(block+1)*16]=delta;plans.append(change)
   for changes in plans:
    branch=copy.deepcopy(env);hist=copy.deepcopy(history);rates=np.maximum(0,base_rate+changes);initial=np.array(hist.rows[-72:],dtype='float32');future=[];cgm=[];bg=[];delivered=[];events=[];terminal=False
    for k in range(48):
     g,b,a,bolus,food,done=step(branch,hist,minute+k*5,rates[k]);future.append(hist.rows[-1]);cgm.append(g);bg.append(b);delivered.append(a);events.append([bolus,food]);terminal=done
     if done:break
    length=len(cgm);features=np.zeros((120,20),dtype='float32');features[:72]=initial;features[72:72+length]=np.array(future);target=np.zeros(48,dtype='float32');target[:length]=cgm;truth=np.zeros(48,dtype='float32');truth[:length]=bg;actions=np.zeros(48,dtype='float32');actions[:length]=delivered;mask=np.arange(48)<length
    arms.append({'features':features,'target':target,'bg':truth,'action':actions,'requested':rates.astype('float32'),'mask':mask,'length':length,'terminal':terminal,'events':np.array(events)})
   for key in ['features','target','bg','action','mask']:np.testing.assert_array_equal(arms[0][key],arms[1][key])
   for arm in arms[2:]:np.testing.assert_array_equal(arm['events'],arms[0]['events'][:len(arm['events'])]) if len(arm['events'])<=len(arms[0]['events']) else np.testing.assert_array_equal(arm['events'][:len(arms[0]['events'])],arms[0]['events'])
   groups.append({'minute':minute,'arms':[arms[0]]+arms[2:]})
  if step(env,history,minute,base_rate)[-1]:base_failure=True;break
 key='p%02d_s%d_b%.1f_r%.1f'%(pid,seed,bolus_factor,basal_factor);out=ROOT/('paired_sim_train_temporal' if temporal else 'paired_sim_train');arrays={}
 for field in ['features','target','bg','action','requested','mask','length','terminal']:
  arrays[field]=np.array([[a[field] for a in g['arms']] for g in groups])
 arrays['minute']=np.array([g['minute'] for g in groups]);path=out/(key+'.npz');np.savez_compressed(path,**arrays)
 return {'file':path.name,'patient':pid,'scenario_seed':seed,'bolus_factor':bolus_factor,'basal_factor':basal_factor,'groups':len(groups),'arms':len(groups)*(7 if temporal else 5),'valid_intervals':int(arrays['mask'].sum()) if groups else 0,'terminated_arms':int(arrays['terminal'].sum()) if groups else 0,'reference_trajectory_terminated':base_failure,'zero_intervention_duplicate_exact':True,'common_meal_bolus_events_exact':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--temporal',action='store_true');args=ap.parse_args()
 out=ROOT/('paired_sim_train_temporal' if args.temporal else 'paired_sim_train');out.mkdir(exist_ok=False);jobs=[(p,s,b,r) for p in [1,2,3,4] for s in [40101,40102] for b in [.8,1.,1.2] for r in [.7,1.,1.3]]
 contract={'purpose':'training-only paired simulator factual arms; identical arrays must be available to the factual-only reference and paired-loss variant','patients':[1,2,3,4],'scenario_seeds':[40101,40102],'training_seed_unchanged':260915,'loop_origins_unchanged':1653421,'development_probe_seed_excluded':3101,'final_scenarios_used':False,'history_features':20,'history_rows':72,'future_steps':48,'action_conditioning':'actual delivered basal U/h; requested rates also retained, no silent action quantization','deltas_u_h':[0,-.5,-.25,.25,.5],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_inputs_exclude_true_bg_and_hidden_physiology':True,'external_FQL_ReBRAC_not_retrained_and_not_fair_same_data_comparators_here':True}
 if args.temporal:
  contract.pop('deltas_u_h');contract.update({'arms_per_group':7,'patterns':'reference and signed0.5 U/h interventions confined to each of three16-step blocks','purpose':'training-only temporal intervention basis; identical factual arrays for all future matched ablations','constant_action_training_replaced_not_overwritten':True,'new_structure_or_efficacy_claim':False})
 (out/'contract.json').write_text(json.dumps(contract,indent=2));records=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for item in pool.map(partial(generate,temporal=args.temporal),jobs):records.append(item);print(json.dumps({'event':'training_scenario_generated',**item}),flush=True)
 result={'status':'complete','contract':contract,'scenarios':records,'groups':sum(r['groups'] for r in records),'arms':sum(r['arms'] for r in records),'valid_intervals':sum(r['valid_intervals'] for r in records),'reference_failures':sum(r['reference_trajectory_terminated'] for r in records)}
 (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['contract','scenarios']}),flush=True)

if __name__=='__main__':main()
