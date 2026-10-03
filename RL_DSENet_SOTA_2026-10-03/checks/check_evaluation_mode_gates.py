"""Fixture-only checks of formal/smoke and completed-update evaluation gates."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(R))
import evaluate_candidates as ev
import world_control_worker as wc
import train_world_policy as tw

checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
def rejects(name,call):
    try:call()
    except (ValueError,FileNotFoundError):checks.append(name)
    else:raise AssertionError('Invalid metadata accepted: '+name)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2))

def main():
    protected=list(R.glob('*.py'))+list((R/'configs').glob('*.json'))
    protected=[p for p in protected if p.name!='evaluate_candidates.py']
    protected+=[p for p in (R/'checks').glob('*.json') if p.name not in ('evaluation_mode_gates.json','evaluation_mode_iql_regression.json')]
    before={str(p):ev.sha(p) for p in protected}
    ast.parse((R/'evaluate_candidates.py').read_text(),feature_version=(3,8))
    check('evaluator_python38_syntax',True)
    protocol=json.loads((R/'protocol.json').read_text())
    with tempfile.TemporaryDirectory(prefix='mode_gate_fixture_',dir=R/'checks') as directory:
        fp=Path(directory);fr=fp/R.name;fb=fp/'RL_DSENet_2026-09-17';fe=fp/'RL_DSENet_公平低糖_2026-09-21'
        copy_paths=[R/n for n in ['protocol.json','train_wide.py','ppo_wide_worker.py','physiologic_features.py','ppo_world_worker.py','train_world_policy.py','world_control_worker.py','world_model_v2.py']]
        copy_paths+=list((R/'configs').glob('*.json'))+ev.COMMON_SOURCES+[ev.NORMALIZER_PATH]
        copy_paths+=[ev.B/n for n in ['world_model.py','policy_bounded.py','forecast_model.py']]
        copy_paths+=[P/'RL_DITR创新_2026-09-16'/n for n in ['response_operator.py','ditr_model.py']]
        for path in copy_paths:
            target=fp/path.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        # Immutable legacy dependency placeholders permit testing the real PPO route.
        weights={}
        for name,relative in [('forecast','results/P03/best.pt'),('world','results/D05_selected_world/world.pt')]:
            p=fb/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fixture_'+name).encode());weights[name]=dict(path=relative,sha256=ev.sha(p))
        context=fp/'context.pt';context.write_bytes(b'fixture_context')
        write(fb/'results/D05_selected_world/config.json',dict(context_checkpoint='context.pt',context_sha256=ev.sha(context)))
        write(fb/'results/D06_selected_policy/config.json',{})
        write(fr/'checks/frozen_selected_version.json',dict(weights=weights))
        world_run=fr/'results/world_fixture';world_run.mkdir(parents=True)
        world_cp=world_run/'best.pt';world_cp.write_bytes(b'fixture_world')
        world_path=world_run/'provenance.json'
        world_training=dict(config=dict(steps=4000,configured_steps=4000,budget_override=False,variant='quantile'))
        def set_world(steps=4000,override=False,configured=4000):
            world_training['config'].update(steps=steps,configured_steps=configured,budget_override=override)
            write(world_path,world_training)
            write(world_run/'completion.json',dict(status='training_completed_candidate_only',steps=steps,configured_steps=configured,budget_override=override))
        set_world()
        count=0
        def make_run(kind,mode,iteration):
            nonlocal count
            count+=1
            prefix='PPO_wide' if kind=='ppo' else 'PPO_world'
            template=fr/'configs'/((('ppo_wide_' if kind=='ppo' else 'ppo_world_')+('risk' if mode=='formal' else 'smoke'))+'.json')
            c=json.loads(template.read_text());c['name']=prefix+'_'+mode+'_'+str(count)
            run=fr/'results'/c['name'];run.mkdir(parents=True)
            if kind=='world_ppo':
                c.update(output_dir=str(run),world_checkpoint=str(world_cp),world_checkpoint_sha256=ev.sha(world_cp),world_training_provenance_sha256=ev.sha(world_path),world_completion_sha256=ev.sha(world_run/'completion.json'))
            cfg=run/'config.json';write(cfg,c)
            if kind=='ppo':
                sources=[fr/'train_wide.py',fr/'ppo_wide_worker.py',fr/'physiologic_features.py',fe/'ppo_env.py']
                train=dict(config=c,training_seed=260915,reward_BG_not_inference_input=True,source_sha256={p.name:ev.sha(p) for p in sources})
            else:
                sources=[fr/n for n in ['ppo_world_worker.py','train_world_policy.py','ppo_wide_worker.py','physiologic_features.py','world_control_worker.py','protocol.json','world_model_v2.py']]+[fp/protocol['scorer']]
                train=dict(config=c,training_seed=260915,protocol_sha256=ev.sha(fr/'protocol.json'),independent_scorer_sha256=ev.sha(fp/protocol['scorer']),world_training_provenance_sha256=ev.sha(world_path),source_sha256={str(p.relative_to(fp)):ev.sha(p) for p in sources})
            write(run/'provenance.json',train)
            cp=run/('policy_iter%02d.pt'%iteration);cp.write_bytes(('fixture policy '+str(iteration)).encode())
            rows=[dict(iteration=i,checkpoint=str(run/('policy_iter%02d.pt'%i)),checkpoint_sha256=ev.sha(cp),transitions=100*i) for i in range(1,iteration+1)]
            (run/'history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return cp,cfg
        def inspect(checkpoint):
            assert Path(checkpoint)==world_cp
            return world_path,deepcopy(world_training),[fr/'world_model_v2.py'],[world_cp,world_path,world_run/'completion.json']
        with patch.multiple(ev,R=fr,P=fp,B=fb,E=fe,SCORER=fp/protocol['scorer'],
                NORMALIZER_PATH=fp/ev.NORMALIZER_PATH.relative_to(P),
                COMMON_SOURCES=[fp/p.relative_to(P) for p in ev.COMMON_SOURCES]),patch.object(tw,'R',fr),patch.object(wc,'inspect_provenance',side_effect=inspect):
            for kind in ['ppo','world_ppo']:
                for iteration in [8,16,32,40]:
                    cp,cfg=make_run(kind,'formal',iteration)
                    binding,_,_,prefix=ev.policy_evaluation_binding(kind,cp,cfg)
                    check(kind+'_formal_iter%02d_allowed'%iteration,binding['iteration']==iteration and binding['checkpoint_sha256']==ev.sha(cp))
                    ev.validate_policy_ready(dict(iteration=iteration),binding,cp,cfg)
                    check(kind+'_ready_iter%02d_matches'%iteration,True)
                    if iteration==8:
                        initial=prefix
                        with (cp.parent/'history.jsonl').open('ab') as f:f.write(b'{incomplete future update')
                        check(kind+'_active_history_tail_does_not_block_or_change_prefix',ev.policy_evaluation_binding(kind,cp,cfg)[3]==initial)
                        check(kind+'_actual_dependencies_accept_formal08',ev.dependencies(kind,cp,cfg)[3]==cp)
                        rejects(kind+'_worker_wrong_iteration_rejected',lambda:ev.validate_policy_ready(dict(iteration=7),binding,cp,cfg))
                        rejects(kind+'_worker_noninteger_iteration_rejected',lambda:ev.validate_policy_ready(dict(iteration=8.),binding,cp,cfg))
                        cp.write_bytes(b'changed checkpoint')
                        rejects(kind+'_checkpoint_changed_after_preflight_rejected',lambda:ev.validate_policy_ready(dict(iteration=8),binding,cp,cfg))
                for iteration in [1,7,10]:
                    cp,cfg=make_run(kind,'formal',iteration)
                    rejects(kind+'_formal_unregistered_iter%02d_rejected'%iteration,lambda:ev.policy_evaluation_binding(kind,cp,cfg))
                    check(kind+'_engineering_smoke_accepts_formal_iter%02d'%iteration,ev.policy_evaluation_binding(kind,cp,cfg,True)[0]['iteration']==iteration)
                cp,cfg=make_run(kind,'smoke',1)
                rejects(kind+'_smoke_training_cannot_be_formal',lambda:ev.dependencies(kind,cp,cfg))
                check(kind+'_explicit_smoke_route_allowed',ev.dependencies(kind,cp,cfg,allow_smoke=True)[3]==cp)
                check(kind+'_smoke_training_mode_recorded',ev.policy_evaluation_binding(kind,cp,cfg,True)[0]['training_mode']=='smoke')
                cp,cfg=make_run(kind,'formal',8);history=cp.parent/'history.jsonl';original=history.read_bytes()
                history.write_bytes(original.rsplit(b'\n',2)[0]+b'\n')
                rejects(kind+'_checkpoint_without_completed_history_row_rejected',lambda:ev.policy_evaluation_binding(kind,cp,cfg))
                history.write_bytes(original[:-1])
                rejects(kind+'_partial_target_history_row_rejected',lambda:ev.policy_evaluation_binding(kind,cp,cfg))
                rows=[json.loads(line) for line in original.splitlines()];rows[-1]['checkpoint_sha256']='0'*64
                history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                rejects(kind+'_history_checkpoint_hash_mismatch_rejected',lambda:ev.policy_evaluation_binding(kind,cp,cfg))
                rows[-1]['checkpoint_sha256']=ev.sha(cp);rows[-1]['checkpoint']=str(cp.parent/'policy_iter07.pt')
                history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                rejects(kind+'_history_checkpoint_path_mismatch_rejected',lambda:ev.policy_evaluation_binding(kind,cp,cfg))
                rows[-1]['checkpoint']=str(cp);rows[2]['iteration']=9
                history.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                rejects(kind+'_noncontiguous_history_rejected',lambda:ev.policy_evaluation_binding(kind,cp,cfg))
            cp,cfg=make_run('ppo','formal',8)
            train_path=cp.parent/'provenance.json';train=json.loads(train_path.read_text());original=deepcopy(train)
            train['config']['copies']=1;write(train_path,train)
            rejects('wide_config_provenance_disagreement_rejected',lambda:ev.policy_evaluation_binding('ppo',cp,cfg))
            train=deepcopy(original);train['source_sha256'].pop('train_wide.py');write(train_path,train)
            rejects('wide_missing_training_source_rejected',lambda:ev.policy_evaluation_binding('ppo',cp,cfg))
            train=deepcopy(original);train['source_sha256']['train_wide.py']='0'*64;write(train_path,train)
            rejects('wide_source_hash_drift_rejected',lambda:ev.policy_evaluation_binding('ppo',cp,cfg))
            # The same budget gate is invoked for standalone MPC and world-PPO.
            control=fr/'configs/world_control_risk.json'
            check('world_formal4000_allowed',ev.dependencies('world',world_cp,control)[3]==world_cp)
            set_world(4,True)
            rejects('MPC_smoke_world_cannot_be_formal',lambda:ev.dependencies('world',world_cp,control))
            check('MPC_smoke_world_explicit_smoke_allowed',ev.dependencies('world',world_cp,control,allow_smoke=True)[3]==world_cp)
            cp,cfg=make_run('world_ppo','smoke',1)
            rejects('worldPPO_smoke_world_policy_cannot_be_formal',lambda:ev.dependencies('world_ppo',cp,cfg))
            check('worldPPO_completed_smoke_world_explicit_smoke_allowed',ev.dependencies('world_ppo',cp,cfg,allow_smoke=True)[3]==cp)
            set_world(3999,False)
            rejects('world3999_even_without_override_refused',lambda:ev.world_evaluation_binding(world_path,world_training))
            set_world(4000,True)
            rejects('world4000_with_override_refused',lambda:ev.world_evaluation_binding(world_path,world_training))
            set_world(4000,False,3999)
            rejects('world_wrong_configured_budget_refused',lambda:ev.world_evaluation_binding(world_path,world_training))
            set_world();completion=json.loads((world_run/'completion.json').read_text());completion['steps']=4;write(world_run/'completion.json',completion)
            rejects('world_completion_config_disagreement_refused',lambda:ev.world_evaluation_binding(world_path,world_training,True))
            cp,cfg=make_run('ppo','formal',8)
            args=SimpleNamespace(kind='ppo',name='ready_mismatch_fixture',checkpoint=cp,config=cfg,
                                 split='development',smoke=True,batch_size=2,method=None)
            class WrongIterationWorker:
                def __init__(self,*unused):self.process=SimpleNamespace(poll=lambda:0,returncode=0)
                def receive(self):return dict(ready=True,iteration=7)
                def close(self):pass
            with patch.object(ev,'JsonWorker',WrongIterationWorker):
                try:ev.run(args)
                except RuntimeError:pass
                else:raise AssertionError('Wrong actual worker iteration accepted')
            out=fr/'results'/args.name
            summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
            binding=manifest['policy_evaluation_binding']
            check('runner_freezes_exact_history_prefix_snapshot',ev.sha(out/manifest['policy_training_history_snapshot'])==binding['history_prefix_sha256'])
            check('runner_records_training_checkpoint_iteration',binding['iteration']==8 and binding['checkpoint_sha256']==ev.sha(cp))
            check('wrong_actual_worker_iteration_retains_technical_failure',summary['status']=='technical_failure' and summary['count']==2)
            check('wrong_iteration_never_starts_simulator_or_fakes_success',all(not item['raw_trajectory_available'] and item['technical_failure_reason'] for item in summary['episodes']))
    check('all_frozen_training_model_worker_IQL_sources_and_existing_checks_unchanged',all(ev.sha(Path(path))==digest for path,digest in before.items()))
    check('no_Torch_or_simulator_loaded',not any(name=='torch' or name.startswith('simglucose') for name in sys.modules))
    result=dict(status='passed',count=len(checks),checks=checks,fixture_only=True,checkpoint_deserialization=False,remote_executed=False,evaluator_sha256=ev.sha(R/'evaluate_candidates.py'),test_source_sha256=ev.sha(Path(__file__).resolve()),protected_source_sha256=before,limitation='Metadata/ready gates only, not real GPU inference or control efficacy. Existing legal smoke and predeclared formal08/16 runs need no rerun.')
    write(R/'checks/evaluation_mode_gates.json',result)
    print(json.dumps(dict(status='passed',count=len(checks),evaluator_sha256=result['evaluator_sha256'])))

if __name__=='__main__':main()
