"""Matched six-hour action interventions; hidden state is only used by simulation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import datetime
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(P/'RL进阶对比_2026-09-15'))
sys.path.insert(0,str(P/'RL_DSENet_公平低糖_2026-09-21'))
from observable_history import History
from evaluate_final import scenario


def plans(anchor):
    out=np.full((9,72),anchor,dtype=np.float64)
    index=1
    for duration in (12,24):
        for multiplier in (0.,.5,1.5,2.):
            out[index,:duration]=anchor*multiplier;index+=1
    return np.clip(out,0,20)


def run(job):
    import pandas as pd
    import pkg_resources
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.simulation.scenario import CustomScenario
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.controller.base import Action
    from controller_baselines import make_controller
    name='adult#%03d'%job['patient']
    params=pd.read_csv(pkg_resources.resource_filename('simglucose','params/vpatient_params.csv'))
    row=params.loc[params.Name==name].iloc[0]
    q=pd.read_csv(pkg_resources.resource_filename('simglucose','params/Quest.csv'))
    cr=float(q.loc[q.Name==name,'CR'].iloc[0]);nominal=float(row.u2ss*row.BW/6000*60)
    meals=scenario(job['seed'],4320);announcements=dict(meals)
    env=T1DSimEnv(T1DPatient.withName(name),CGMSensor.withName('GuardianRT',seed=job['seed']+10000),
                 InsulinPump.withName('Insulet'),CustomScenario(datetime.datetime(2020,1,1),[(m/60,g) for m,g in meals]))
    initial=env.reset();assert env.sample_time==5
    history=History();history.append(0,float(initial.observation.CGM))
    controller=make_controller('explore',seed=260915+job['index'])
    anchor=None;groups=[];capture=set(range(360,3961,480));copy_checks=0;base_terminal=False
    if job['smoke']:capture={360}
    for minute in range(0,4320,5):
        if minute<360:requested=nominal
        else:
            state=history.state()
            if anchor is None:anchor=float((state[-1,1]*.14462788945609448+.09945811581924525)*12)
            if minute in capture:
                arm_actions=plans(anchor);cgms=[];bgs=[];actuals=[];masks=[];foods=[];boluses=[];terminals=[]
                # Smoke repeats the reference from a separately copied state.
                run_plans=list(arm_actions)+([arm_actions[0]] if job['smoke'] else [])
                for arm_index,plan in enumerate(run_plans):
                    branch=copy.deepcopy(env);cgm=[];bg=[];actual=[];food=[];boluses_arm=[];terminal=False
                    for step,rate in enumerate(plan):
                        when=minute+step*5
                        bolus=job['bolus_factor']*announcements.get(when,0)/cr
                        delivered=float(branch.pump.basal(float(rate)/60))*60
                        delivered_bolus=float(branch.pump.bolus(bolus/5))*5
                        previous=branch.time
                        transition=branch.step(Action(basal=float(rate)/60,bolus=bolus/5))
                        assert (branch.time-previous).total_seconds()/60==5
                        assert abs(float(branch.insulin_hist[-1])*5-(delivered/12+delivered_bolus))<1e-8
                        cgm.append(float(transition.observation.CGM));bg.append(float(transition.info['bg']));actual.append(delivered)
                        food.append(float(transition.info['meal'])*5);boluses_arm.append(delivered_bolus)
                        if not np.isfinite(cgm[-1]) or not np.isfinite(bg[-1]):raise ValueError('Nonfinite branch label')
                        if transition.done:terminal=True;break
                    mask=np.arange(72)<len(cgm)
                    if arm_index==9:
                        assert np.array_equal(cgms[0][:len(cgm)],cgm)
                        assert np.array_equal(bgs[0][:len(bg)],bg)
                        assert np.array_equal(actuals[0][:len(actual)],actual)
                        assert np.array_equal(masks[0],mask)
                        assert np.array_equal(foods[0][:len(food)],food)
                        assert np.array_equal(boluses[0][:len(boluses_arm)],boluses_arm)
                        copy_checks+=1
                        continue
                    if foods:
                        common=min(int(np.sum(masks[0])),len(food))
                        assert np.array_equal(foods[0][:common],food[:common])
                        assert np.array_equal(boluses[0][:common],boluses_arm[:common])
                    cgms.append(cgm+[0.]*(72-len(cgm)));bgs.append(bg+[0.]*(72-len(bg)))
                    actuals.append(actual+[anchor]*(72-len(actual)));masks.append(mask)
                    foods.append(food+[0.]*(72-len(food)));boluses.append(boluses_arm+[0.]*(72-len(boluses_arm)))
                    terminals.append(terminal)
                groups.append(dict(history=state,anchor_u_h=anchor,actions_u_h=actuals,
                                   target_cgm_mg_dl=cgms,target_bg_mg_dl=bgs,mask=masks,origin_minute=minute,
                                   label_only_future_meal_g=foods,label_only_future_bolus_u=boluses,
                                   arm_native_terminal=terminals,arm_steps=np.sum(masks,axis=1)))
            requested=float(controller.action(state[None],np.array([anchor]))[0])
        bolus=job['bolus_factor']*announcements.get(minute,0)/cr
        basal=float(env.pump.basal(requested/60));b=float(env.pump.bolus(bolus/5))
        duplicate=copy.deepcopy(env) if job['smoke'] and minute==360 else None
        previous=env.time
        transition=env.step(Action(basal=requested/60,bolus=bolus/5))
        assert (env.time-previous).total_seconds()/60==5
        assert abs(float(env.insulin_hist[-1])*5-(basal+b)*5)<1e-8
        if duplicate is not None:
            same=duplicate.step(Action(basal=requested/60,bolus=bolus/5))
            assert same.observation.CGM==transition.observation.CGM and same.info['bg']==transition.info['bg']
            assert duplicate.insulin_hist[-1]==env.insulin_hist[-1] and same.done==transition.done
            copy_checks+=1
        history.append(minute+5,float(transition.observation.CGM),basal*5,
                       b*5 if b>0 else None,float(transition.info['meal'])*5 if transition.info['meal']>0 else None)
        if not np.isfinite(transition.observation.CGM) or not np.isfinite(transition.info['bg']):
            raise ValueError('Nonfinite base trajectory')
        if transition.done:base_terminal=True;break
        if job['smoke'] and minute>=360:break
    if not groups:raise ValueError('No complete-history intervention origin')
    arrays={key:np.asarray([g[key] for g in groups],dtype=bool if key in ('mask','arm_native_terminal') else np.float32) for key in groups[0]}
    path=R/'cache'/job['name']/('p%02d_s%d_b%.1f.npz'%(job['patient'],job['seed'],job['bolus_factor']))
    np.savez_compressed(path,**arrays)
    return dict(file=path.name,groups=len(groups),arms=9*len(groups),copy_checks=copy_checks,job=job,
                base_native_terminal=base_terminal,base_last_minute=minute+5,
                base_administrative_smoke_stop=bool(job['smoke'] and not base_terminal),
                arm_native_terminal_count=int(arrays['arm_native_terminal'].sum()),
                valid_branch_points=int(arrays['mask'].sum()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked_run(job):
    try:return run(job)
    except Exception as error:
        path=R/'cache'/job['name']/('failed_job_%03d.json'%job['index'])
        path.write_text(json.dumps(dict(job=job,error=repr(error),status='failed'),indent=2))
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--split',choices=['train','world_validation'],required=True)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=R/'cache'/a.name;out.mkdir(parents=True,exist_ok=False)
    p=json.loads((R/'protocol.json').read_text());jobs=[]
    for patient in p['patients']:
        for seed in p[a.split+'_scenario_seeds'][:2]:
            for bf in p['bolus_factors']:
                jobs.append(dict(patient=patient,seed=seed,bolus_factor=bf,index=len(jobs),name=a.name,smoke=a.smoke))
    if a.smoke:jobs=[jobs[0],next(j for j in jobs if j['patient']==9 and j['bolus_factor']==1.2)]
    source={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),R/'protocol.json',R/'controller_baselines.py',R/'physiologic_features.py',
             P/'RL进阶对比_2026-09-15/observable_history.py',P/'RL_DSENet_公平低糖_2026-09-21/evaluate_final.py',
             P/'Loop数据集/训练管线_v2/prepared/normalization.json']}
    (out/'started.json').write_text(json.dumps(dict(jobs=jobs,source_sha256=source),indent=2))
    records=[]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for result in pool.imap_unordered(checked_run,jobs):
            records.append(result);print(json.dumps(result),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',kind='paired_interventions',split=a.split,
                                                     horizon=72,arms=9,episodes=records,source_sha256=source,
                                                     hidden_simulator_state_is_not_model_input=True),indent=2))


if __name__=='__main__':main()
