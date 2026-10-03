"""Mean-deployment eval checks; metadata fixtures/artificial logits, no GPU."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(R))
import evaluate_candidates as ev
import ppo_mean_worker as mean

checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
def reject(name,call):
    try:call()
    except ValueError:checks.append(name)
    else:raise AssertionError('Invalid mean interface accepted: '+name)

def main():
    protected=[p for p in R.glob('*.py') if p.name!='evaluate_candidates.py']
    protected+=list((R/'configs').glob('*.json'))
    protected+=[p for p in (R/'checks').glob('*.json') if p.name!='mean_evaluation_mechanics.json']
    original_hashes={str(p):ev.sha(p) for p in protected}
    ast.parse((R/'evaluate_candidates.py').read_text(),feature_version=(3,8))
    check('python38_syntax',True)
    h=np.zeros((2,72,22),np.float32);anchors=np.array([.923457182,20.])
    for kind in ev.MEAN_KINDS:
        check(kind+'_only_observable_evaluate_payload',ev.policy_payload(kind,h,anchors)==dict(op='evaluate',history=h.tolist(),anchors=anchors.tolist()))
    logits=[[0.,1.,2.,3.,4.,3.,2.,1.,0.],[0.,-1000.,-1000.,-1000.,-1000.,-1000.,-1000.,-1000.,0.]]
    response=mean.distribution_summary(logits,anchors.astype(np.float32).astype(float).tolist(),mean.RATIOS)
    actions=np.array(response['actions'])
    diagnostics=ev.validate_mean_response(response,actions,anchors)
    check('independent_probabilities_grid_mean_validation',len(diagnostics)==2)
    check('cap_each_action_before_mean',abs(actions[1]-10)<1e-12 and abs(response['raw_ratio_mean_rate'][1]-20)<1e-12)
    for kind in ev.MEAN_KINDS:
        check(kind+'_continuous_mean_accepted',np.array_equal(ev.validate_actions(kind,[.123,10.123],anchors),[.123,10.123]))
    reject('old_world_argmax_grid_gate_unchanged',lambda:ev.validate_actions('world_ppo',[.123,10.123],anchors))
    logs=[[math.log(x) if x else -1000 for x in p] for p in [[.3,0,0,0,.6,0,0,0,.1],[.1,0,0,0,.6,0,0,0,.3]]]
    same=mean.distribution_summary(logs,[1.,1.],mean.RATIOS)
    ev.validate_mean_response(same,np.array(same['actions']),np.array([1.,1.]))
    check('same_argmax_can_have_distinct_means',same['argmax_indices']==[4,4] and np.allclose(same['actions'],[.8,1.2]))
    for field,value,label in [
        ('probabilities',[[1.]*9]*2,'unnormalized_probabilities'),
        ('probabilities',[[-.1]+[.1375]*8]*2,'negative_probability'),
        ('probabilities',[[float('nan')]*9]*2,'nonfinite_probability'),
        ('probabilities',[[1/8]*8]*2,'wrong_probability_shape'),
        ('actual_action_grid_u_h',[[0.]*9]*2,'wrong_executable_grid'),
        ('action_mean',[1.,10.],'duplicate_mean_mismatch'),
        ('argmax_indices',[9,0],'argmax_outside_grid'),
        ('argmax_indices',[0.,0.],'noninteger_argmax'),
        ('argmax_indices',[0,0],'nonmaximum_argmax'),
        ('entropy',[0.,0.],'entropy_mismatch'),
        ('arithmetic_boundary_adjustment_u_h',[1e-8,0.],'nontrivial_boundary_adjustment')]:
        bad=deepcopy(response);bad[field]=value
        reject(label,lambda bad=bad:ev.validate_mean_response(bad,actions,anchors))
    bad=deepcopy(response);bad['actual_action_grid_u_h'][1][-1]=40
    reject('uncapped_grid_rejected',lambda:ev.validate_mean_response(bad,actions,anchors))
    bad=deepcopy(response);bad['actions'][1]=20;bad['action_mean'][1]=20
    reject('mean_ratio_then_cap_rejected',lambda:ev.validate_mean_response(bad,np.array(bad['actions']),anchors))
    ready_common=dict(kind='ppo_mean',iteration=16,deployment_rule=ev.MEAN_RULE,original_deployment_rule='argmax',
        training_enabled=False,rollout_buffer_retained=False,random_action_sampling=False,original_preregistration=False,
        same_checkpoint_new_deployment_ablation=True,proposed_after_ppo08_development_results=True)
    binding=dict(checkpoint_sha256='a'*64,config_sha256='b'*64,training_provenance_sha256='c'*64,history_prefix_sha256='d'*64,history_prefix_bytes=1600,iteration=16)
    for kind,family,dim,old in [('ppo_mean','wide',334,'ppo_wide_worker.py'),('world_ppo_mean','world',195,'ppo_world_worker.py')]:
        source_hashes={str((R/n).relative_to(P)):ev.sha(R/n) for n in ['ppo_mean_worker.py',old]}
        manifest=dict(policy_evaluation_binding=binding,source_sha256=source_hashes)
        ready=dict(ready_common,**binding,family=family,feature_dim=dim,source_sha256=source_hashes)
        ev.validate_mean_ready(ready,kind,manifest);check(kind+'_strict_ready_passes',True)
        for field,value in [('family','wrong'),('deployment_rule','argmax'),('history_prefix_sha256','0'*64),('checkpoint_sha256','0'*64),('source_sha256',{}),('training_enabled',True),('original_preregistration',True),('proposed_after_ppo08_development_results',False)]:
            bad=dict(ready);bad[field]=value
            reject(kind+'_ready_'+field+'_rejected',lambda bad=bad:ev.validate_mean_ready(bad,kind,manifest))
    # Reuse all established family budget/history gates with dependency calls
    # routed through the new aliases, retaining the original check artifacts.
    spec=importlib.util.spec_from_file_location('mode_gates',R/'checks/check_evaluation_mode_gates.py')
    mode=importlib.util.module_from_spec(spec);spec.loader.exec_module(mode)
    base_dependencies=ev.dependencies;depth=0;commands={};mode_result={}
    def routed(kind,*args,**kwargs):
        nonlocal depth
        if depth==0 and kind in ('ppo','world_ppo'):kind={'ppo':'ppo_mean','world_ppo':'world_ppo_mean'}[kind]
        depth+=1
        try:
            answer=base_dependencies(kind,*args,**kwargs)
            if kind in ev.MEAN_KINDS:commands[kind]=answer[2]
            return answer
        finally:depth-=1
    base_run=ev.run
    def run(args):
        if args.kind=='ppo':args.kind='ppo_mean'
        try:return base_run(args)
        finally:
            path=ev.R/'results'/args.name/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text())
                check('runner_mean_is_postdevelopment_same_weight_ablation',m.get('deployment_rule')==ev.MEAN_RULE and m.get('original_preregistration') is False and m.get('new_external_method') is False and m.get('new_training') is False)
    original_write=mode.write
    def capture(path,value):
        if path.name=='evaluation_mode_gates.json':mode_result.update(value)
        else:original_write(path,value)
    with patch.object(ev,'dependencies',routed),patch.object(ev,'run',run),patch.object(ev,'COMMON_SOURCES',ev.COMMON_SOURCES+[R/'ppo_mean_worker.py']),patch.object(mode,'write',capture):mode.main()
    check('all73_underlying_mode_history_ready_checks_pass_through_mean_routes',mode_result['count']==73 and mode_result['status']=='passed')
    for kind,command in commands.items():
        family='wide' if kind=='ppo_mean' else 'world'
        check(kind+'_worker_family_and_original_run_config',command[2].endswith('/ppo_mean_worker.py') and command[3:5]==['--family',family] and command[5]=='--config' and command[6].endswith('/config.json') and command[7]=='--checkpoint')
    import check_iql_evaluation as iql
    iql_result={};iql_write=iql.write
    def capture_iql(path,value):
        if path.name=='evaluate_iql_wide_mechanics.json':iql_result.update(value)
        else:iql_write(path,value)
    with patch.object(iql,'write',capture_iql):iql.main()
    check('existing58_IQL_checks_still_pass',iql_result['count']==58 and iql_result['status']=='passed')
    check('all_training_original_workers_configs_and_prior_checks_unchanged',all(ev.sha(Path(p))==digest for p,digest in original_hashes.items()))
    check('no_Torch_or_simulator_loaded',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    result=dict(status='passed',count=len(checks),checks=checks,inherited_family_gate_checks=mode_result['checks'],inherited_family_gate_count=mode_result['count'],IQL_regression_checks=iql_result['checks'],IQL_regression_count=iql_result['count'],fixture_only=True,artificial_logits_only=True,checkpoint_deserialization=False,remote_executed=False,evaluator_sha256=ev.sha(R/'evaluate_candidates.py'),mean_worker_sha256=ev.sha(R/'ppo_mean_worker.py'),test_source_sha256=ev.sha(Path(__file__).resolve()),original_source_and_check_hashes=original_hashes,limitation='No real CUDA checkpoint inference or closed-loop result. Same-checkpoint deployment rule proposed after PPO08 development; not original preregistration or a new external method.')
    (R/'checks/mean_evaluation_mechanics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='passed',new_checks=len(checks),inherited_checks=mode_result['count'],evaluator_sha256=result['evaluator_sha256'])))

if __name__=='__main__':main()
