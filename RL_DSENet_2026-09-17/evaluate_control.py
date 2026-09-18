"""Independent public simulator processes, batched observable-only Torch requests."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
import argparse,datetime,hashlib,json,multiprocessing as mp,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RL进阶对比_2026-09-15'))
sys.path.insert(0,str(ROOT.parent/'RL_DITR创新_2026-09-16'))
from observable_history import History
from control_metrics import summarize

def scenario(seed,total_minutes,irregular=False):
 rng=np.random.default_rng(seed);meals=[]
 for day in range((total_minutes+1439)//1440):
  for hour,grams in [(8,45),(13,65),(19,70)]:
   jitter=18 if irregular else 6;sd=20 if irregular else 10
   minute=day*1440+hour*60+int(rng.integers(-jitter,jitter+1))*5;amount=float(np.clip(grams+rng.normal(0,sd),20,120))
   if minute<total_minutes:meals.append([minute,amount])
 return meals

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
 connection.send({'done':True,'result':result});connection.close()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bounded-reference',action='store_true');ap.add_argument('--config',required=True);ap.add_argument('--checkpoint');ap.add_argument('--mode',choices=['actor','beam','planner','hold','nominal','legacy','ditr'],default='beam');ap.add_argument('--horizon',type=int,default=48);ap.add_argument('--block-steps',type=int,default=16);ap.add_argument('--name',required=True);ap.add_argument('--batch-size',type=int,default=16);args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text());out=ROOT/'results'/args.name;out.mkdir(parents=True,exist_ok=False);ctx=mp.get_context('spawn');jobs=[]
 for group in cfg['groups']:
  for p in cfg['patients']:
   for seed in group['seeds']:
    jobs.append({'patient':p,'seed':seed,'group':group['name'],'bolus_factor':group['bolus_factor'],'total_minutes':cfg['total_minutes'],'meals':scenario(seed,cfg['total_minutes'],group.get('irregular',False))})
 manifest={'bounded_reference':args.bounded_reference,'config':cfg,'mode':args.mode,'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'jobs':jobs,'comparison_status':cfg['comparison_status'],'baseline_retraining':False,'frozen_baseline_reevaluation':args.mode in ('legacy','ditr'),'seed_is_scenario_not_training':True}
 if args.mode=='chunk':manifest['planning']={'horizon_steps':args.horizon,'block_steps':args.block_steps,'beam_size':10,'full_5min_reward_path':True}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False));server=None;all_results=[];all_latencies=[];start=time.time();hold_rates={}
 if args.mode not in ('nominal','hold'):
  if not args.checkpoint:raise ValueError('Checkpoint required')
  command=[str(ROOT.parent/'.venv-native/bin/python'),str(ROOT/('legacy_policy_worker.py' if args.mode=='legacy' else ('bounded_policy_worker.py' if args.bounded_reference else 'new_policy_worker.py'))),'--checkpoint',str(Path(args.checkpoint).resolve()),'--mode',args.mode]
  if args.mode=='ditr':command=[str(ROOT.parent/'.venv-native/bin/python'),str(ROOT.parent/'RL_DITR创新_2026-09-16/policy_worker.py'),'--checkpoint',str(Path(args.checkpoint).resolve()),'--mode','beam']
  if args.mode=='chunk':command+=['--horizon',str(args.horizon),'--block-steps',str(args.block_steps)]
  server=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
  ready=json.loads(server.stdout.readline());assert ready.get('ready'),ready
 try:
  for offset in range(0,len(jobs),args.batch_size):
   active=[]
   for job in jobs[offset:offset+args.batch_size]:
    parent,child=ctx.Pipe();process=ctx.Process(target=environment_worker,args=(child,job));process.start();child.close();active.append((parent,process,job))
   iterations=0;latencies=[]
   while active:
    pending=[];states=[]
    for conn,process,job in active:
     if not conn.poll(120):raise TimeoutError('Simulator worker stalled: '+str(job))
     message=conn.recv()
     if message['done']:
      process.join(10);result=message['result'];key='%s_p%02d_s%d'%(job['group'],job['patient'],job['seed']);(out/(key+'.json')).write_text(json.dumps(result,allow_nan=False));all_results.append({'key':key,'patient':job['patient'],'group':job['group'],'seed':job['seed'],'failure_reason':result['failure_reason'],'metrics':result['metrics']});conn.close()
      print(json.dumps({'event':'episode_complete','name':args.name,'key':key,'failed':result['metrics']['failed'],'tir_bounds':[result['metrics']['bg']['tir_lower_bound_pct'],result['metrics']['bg']['tir_upper_bound_pct']]}),flush=True)
     else:pending.append((conn,process,job,message['nominal_u_h']));states.append(message['history'])
    active=[(c,p,j) for c,p,j,_ in pending]
    if not pending:break
    if args.mode=='nominal':actions=[n for _,_,_,n in pending]
    elif args.mode=='hold':
     actions=[]
     for (_,_,job,_),state in zip(pending,states):
      key=(job['patient'],job['group'],job['seed'])
      if key not in hold_rates:
       assert state[-1,6]>.5
       hold_rates[key]=float((state[-1,1]*0.14462788945609448+0.09945811581924525)*12)
      actions.append(hold_rates[key])
    else:
     payload={'history':np.stack(states).tolist()}
     if args.mode=='legacy':
      payload['case_keys']=[str((j['patient'],j['group'],j['seed'])) for _,_,j,_ in pending]
      payload['scenario_seeds']=[j['seed'] for _,_,j,_ in pending]
     if args.bounded_reference:
      anchors=[]
      for (_,_,job,_),state in zip(pending,states):
       key=(job['patient'],job['group'],job['seed'])
       if key not in hold_rates:
        assert state[-1,6]>.5
        hold_rates[key]=float((state[-1,1]*0.14462788945609448+0.09945811581924525)*12)
       anchors.append(hold_rates[key])
      payload['anchor_u_h']=anchors
     server.stdin.write(json.dumps(payload)+'\n');server.stdin.flush();response=json.loads(server.stdout.readline())
     if 'error' in response:
      for conn,_,_,_ in pending:conn.send({'abort':response['error']})
      continue
     actions=response['actions_u_h'];latencies.append(response['batch_seconds']);all_latencies.append({'batch_size':len(pending),'seconds':response['batch_seconds']});assert len(actions)==len(pending)
    for (conn,_,_,_),action in zip(pending,actions):conn.send({'action_u_h':action})
    iterations+=1
    if iterations%144==0:print(json.dumps({'event':'progress','name':args.name,'offset':offset,'decisions_per_active_episode':iterations,'active':len(active),'elapsed_seconds':time.time()-start}),flush=True)
  timing={'scope':'GPU worker batch computation, excludes JSON transport and simulator; shared GPU load may vary','batches':all_latencies}
  (out/'inference_timing.json').write_text(json.dumps(timing,allow_nan=False))
  (out/'summary.json').write_text(json.dumps({'episodes':all_results,'count':len(all_results),'wall_seconds':time.time()-start,'status':'completed','training_seed':260915},indent=2,allow_nan=False))
 finally:
  if server is not None:server.stdin.close();server.wait(timeout=30)
  for conn,process,_ in locals().get('active',[]):
   if process.is_alive():process.terminate();process.join()

if __name__=='__main__':main()
