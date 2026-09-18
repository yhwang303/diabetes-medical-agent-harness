"""Freeze selected weights, evaluation jobs and implementation before final access."""
import datetime
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    selection=json.loads((ROOT/'checks/patch_selection.json').read_text())
    forecast=ROOT/'results'/selection['selected']/'best.pt'
    world=ROOT/'results/D05_selected_world/world.pt';policy=ROOT/'results/D06_selected_policy/policy.pt'
    assert json.loads((world.parent/'development_diagnostics.json').read_text())['decision_gate_passed']
    assert json.loads((policy.parent/'completion.json').read_text())['world_frozen_exact']
    pcfg=json.loads((policy.parent/'config.json').read_text());wcfg=json.loads((world.parent/'config.json').read_text())
    assert pcfg['world_sha256']==sha(world) and wcfg['forecast_sha256']==sha(forecast)
    # Development outputs must exist before final access; no validation-score checkpoint tuning here.
    for name in ('E08_selected_planner_dev','E09_selected_actor_dev','E10_selected_beam_dev'):
        assert json.loads((ROOT/'results'/name/'summary.json').read_text())['status']=='completed'
    assert json.loads((ROOT/'checks/frozen_baseline_workers.json').read_text())['status']=='passed'
    spec=[('F01_hold','Fixed observed basal','hold',None,'observed warmup basal; no training')]
    old=ROOT.parent/'RL进阶对比_2026-09-15'
    frozen=json.loads((old/'configs/T01_frozen_sealed_comparison.json').read_text())
    for index,key,label in [(2,'BC','BC'),(3,'ReBRAC_BC100','ReBRAC'),(4,'FQL_alpha100','FQL'),(5,'Anchor_FQL','Anchor-FQL'),(6,'Ours_bound040','Previous bounded RL'),(7,'Bound040_noRL','Previous bounded no-RL')]:
        path=old/frozen['weights'][key]['path'];assert sha(path)==frozen['weights'][key]['sha256']
        spec.append(('F%02d_%s'%(index,key),label,'legacy',path,'Loop; frozen prior weights, original action contract'))
    ditr=ROOT.parent/'RL_DITR创新_2026-09-16'
    spec += [('F08_R05','RL-DITR adaptation R05','ditr',ditr/'results/R05_recursive_sim_stable_policy/policy_last.pt','Loop + prior simulation; original 1h planning'),
             ('F09_H02','RL-DITR adaptation H02','ditr',ditr/'results/H02_prefix_sim_factual/policy_last.pt','Loop + prior simulation; original 1h planning'),
             ('F10_planner','DSENet world + no-RL planner','planner',world,'Loop + H02 context + 504 paired simulation groups'),
             ('F11_actor','DSENet world + RL policy (ours)','actor',policy,'same world and action set as no-RL planner'),
             ('F12_beam','DSENet world + RL + planning','beam',policy,'same world and action set as no-RL planner')]
    methods=[]
    for name,label,mode,path,info in spec:
        methods.append({'name':name,'label':label,'mode':mode,'path':str(path.relative_to(ROOT.parent)) if path else None,'sha256':sha(path) if path else None,'bounded_reference':mode in ('planner','actor','beam'),'training_information':info})
    config={'name':'final_frozen_new_scenarios','patients':list(range(1,11)),'total_minutes':4320,
            'groups':[{'name':name,'bolus_factor':factor,'seeds':[91701,91702],'irregular':False} for name,factor in [('normal',1.),('under_bolus',.8),('over_bolus',1.2)]],
            'comparison_status':'frozen new scenarios on previously exposed virtual adults; not unseen-patient test'}
    output={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_further_selection':True,'training_seed':260915,'primary_method':'F11_actor',
            'patients':config['patients'],'episode_count':60,'forecast':{'path':str(forecast.relative_to(ROOT)),'sha256':sha(forecast)},
            'world':{'path':str(world.relative_to(ROOT)),'sha256':sha(world)},'policy':{'path':str(policy.relative_to(ROOT)),'sha256':sha(policy)},
            'patch_selection_sha256':sha(ROOT/'checks/patch_selection.json'),'methods':methods,'scenario_config':config,
            'primary_outcomes':['patient-equal BG TIR missing-outcome bounds','observed TBR70/54','failure and coverage'],
            'secondary_outcomes':['CGM counterparts','TAR','CV','risk','insulin','action variation'],
            'inference':'10000 paired patient-cluster bootstrap; 10 independent virtual patients, not 60; no training-seed uncertainty',
            'claim_rule':'report superiority only for measured endpoints; increased TBR must accompany TIR gains; no clinical or SOTA claim',
            'source_sha256':{str(p.relative_to(ROOT.parent)):sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT/'dsenet').rglob('*.py'))+list(ditr.glob('*.py'))+[old/'numpy_policy.py',old/'observable_history.py',ROOT.parent/'RL训练_2026-09-15/data.py',ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json'] if p.exists()}}
    (ROOT/'configs/final_scenarios.json').write_text(json.dumps(config,indent=2))
    target=ROOT/'configs/final_freeze.json'
    with target.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps({'freeze_sha256':sha(target),'forecast':selection['selected'],'methods':len(methods),'episodes_each':60,'no_further_selection':True}))

if __name__=='__main__':main()
