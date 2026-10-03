"""Local IQL evaluation metadata/action mechanics; no Torch or simulator run."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import evaluate_candidates as ev
import prepare_iql_replay as prep

R=Path(__file__).resolve().parent; P=R.parent
CHECKS=[]
def check(name,condition):
    assert condition,name
    CHECKS.append(name)
def rejected(name,call):
    try: call()
    except (ValueError,FileNotFoundError,FileExistsError): CHECKS.append(name)
    else: raise AssertionError('Accepted invalid fixture: '+name)
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2))

def main():
    immutable=[p for p in R.glob('*.py') if p.name not in ('evaluate_candidates.py','check_iql_evaluation.py')]
    immutable+=list((R/'configs').glob('*.json'))
    prior_checks=[R/'checks'/n for n in ('evaluate_candidates_mechanics.json','evaluate_world_ppo_metadata_checks.json','retained_evaluation_mechanics.json')]
    before={str(p):ev.sha(p) for p in immutable+prior_checks}
    source=R/'evaluate_candidates.py'; ast.parse(source.read_text()); compile(source.read_text(),str(source),'exec')
    check('evaluator_compiles',True)
    protocol=json.loads((R/'protocol.json').read_text())
    jobs=ev.make_jobs(protocol,'development')
    check('same60jobs',len(jobs)==60 and ev.json_sha(jobs)=='92c50ab6c353023716a5e3fc7bfa76f0ac0999c9363f3cb82e0a5eaf0ed4a417')
    rejected('confirmation_refused',lambda:ev.make_jobs(protocol,'confirmation'))
    check('same_world_validation60jobs',len(ev.make_jobs(protocol,'world_validation'))==60)
    h=np.zeros((2,72,22),np.float32); a=np.array([1.,12.])
    payload=ev.policy_payload('iql_wide',h,a)
    check('IQL_observable_evaluate_only_payload',payload==dict(op='evaluate',history=h.tolist(),anchors=a.tolist()))
    for kind in ('ppo','world','world_ppo'):
        check(kind+'_payload_unchanged',ev.policy_payload(kind,h,a)==payload)
    check('legacy_payload_unchanged',ev.policy_payload('legacy',h,a)==dict(history=h.tolist(),anchor_u_h=a.tolist()))
    check('retained_DITR_payload_unchanged',ev.policy_payload('retained',h,a,method='ditr')==dict(history=h.tolist()))
    check('retained_projection_unchanged',np.array_equal(ev.project_retained_actions([10.,20.],[1.,12.])[0],[2.,20.]))
    check('IQL_continuous_action_not_grid_limited',np.array_equal(ev.validate_actions('iql_wide',[.123,19.123],a),[.123,19.123]))
    rejected('IQL_out_of_support_rejected',lambda:ev.validate_actions('iql_wide',[2.1,20.],a))
    normalized=np.array([-.877,1.]); actions=np.minimum(a*(normalized+1),20.)
    response=dict(actions=actions.tolist(),actions_u_h=actions.tolist(),normalized_actions=normalized.tolist(),global_cap_applied=[False,True])
    check('IQL_normalized_intrinsic_cap_matches',np.array_equal(ev.validate_iql_response(response,actions,a)[0],normalized))
    for name,field,value in [('duplicate_rate_mismatch','actions_u_h',[1.,20.]),('normalized_outside_range','normalized_actions',[-1.1,1.]),('nonfinite_normalized','normalized_actions',[float('nan'),1.]),('wrong_cap_flag','global_cap_applied',[False,False]),('integer_cap_flag','global_cap_applied',[0,1])]:
        bad=dict(response);bad[field]=value
        rejected(name,lambda bad=bad:ev.validate_iql_response(bad,actions,a))
    config=prep.load_config((R/'configs/iql_wide.json').resolve())
    original_required=prep.source_paths(config)+[R/'configs/iql_wide.json']
    with tempfile.TemporaryDirectory(prefix='iql_eval_fixture_',dir=R/'checks') as tmp:
        fp=Path(tmp);fr=fp/R.name
        paths=[]
        for actual in set(original_required+ev.COMMON_SOURCES):
            path=fp/actual.relative_to(P);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(actual.read_bytes() if actual.is_file() else b'fixture dependency, not a runtime model\n')
            paths.append(path)
        template=fr/'configs/iql_wide.json';write(template,config)
        write(fr/'protocol.json',protocol)
        required=[fp/p.relative_to(P) for p in original_required]
        run=fr/'results/IQL_wide';run.mkdir(parents=True)
        cp=run/'policy_020000.pt';cp.write_bytes(b'fixture checkpoint; never torch loaded')
        cfg=run/'config.json';write(cfg,config)
        replay=fr/config['replay'];replay.mkdir(parents=True)
        source_hashes={str(p.relative_to(fp)):ev.sha(p) for p in required}
        replay_data=dict(status='complete',config=config,source_sha256=source_hashes,raw_data_sha256={'fixture_raw':'f'*64},selection='fixture declared train selection',episodes=280)
        write(replay/'manifest.json',replay_data)
        provenance=dict(smoke=False,training_seed=260915,planned_updates=20000,config_sha256=ev.sha(template),source_sha256=source_hashes,
            replay_path=str(replay.relative_to(fp)),replay_manifest_sha256=ev.sha(replay/'manifest.json'),raw_data_sha256=replay_data['raw_data_sha256'],data_selection=replay_data['selection'],episodes=280)
        completion=dict(status='fixed_budget_completed',steps=20000,seed=260915,sample_visits=20000*256,final_checkpoint=cp.name,development_or_confirmation_used=False,checkpoints=[dict(path=cp.name,step=20000,sha256=ev.sha(cp))])
        def reset():
            write(cfg,config);write(run/'provenance.json',provenance);write(run/'completion.json',completion);write(replay/'manifest.json',replay_data)
        reset()
        with patch.multiple(ev,P=fp,R=fr,SCORER=fp/protocol['scorer'],NORMALIZER_PATH=fp/ev.NORMALIZER_PATH.relative_to(P),COMMON_SOURCES=[fp/p.relative_to(P) for p in ev.COMMON_SOURCES]),patch.object(prep,'source_paths',lambda c:required[:-1]):
            c,b,sources,artifacts=ev.iql_definition(cp,cfg)
            check('fixed_final_metadata_source_binding',c==config and b==provenance and set(required).issubset(sources))
            ss,aa,cmd,returned=ev.dependencies('iql_wide',cp,cfg)
            check('original_IQL_worker_cuda_command',cmd==[str(fp/'.venv-native/bin/python'),'-u',str(fr/'iql_wide_worker.py'),'--checkpoint',str(cp),'--device','cuda'])
            check('weight_config_provenance_completion_replay_hashed',all(p in aa for p in [cp,cfg,run/'provenance.json',run/'completion.json',replay/'manifest.json']))
            check('no_old_D05_H02_weight_dependencies',not any('D05' in str(p) or 'H02' in str(p) for p in aa))
            rejected('relative_checkpoint_refused',lambda:ev.iql_definition(Path('policy_020000.pt'),cfg))
            rejected('relative_config_refused',lambda:ev.iql_definition(cp,Path('config.json')))
            rejected('different_run_refused',lambda:ev.iql_definition(run.parent/'other/policy_020000.pt',cfg))
            rejected('intermediate_checkpoint_refused',lambda:ev.iql_definition(run/'policy_005000.pt',cfg))
            bad=dict(config);bad['expectile']=.8;write(cfg,bad)
            rejected('run_config_drift_refused',lambda:ev.iql_definition(cp,cfg));reset()
            for label,field,value in [('wrong_training_seed','training_seed',1),('wrong_budget','planned_updates',19999),('template_hash_mismatch','config_sha256','0'*64),('replay_hash_mismatch','replay_manifest_sha256','0'*64),('outside_replay_path','replay_path','../outside')]:
                bad=deepcopy(provenance);bad[field]=value;write(run/'provenance.json',bad)
                rejected(label,lambda:ev.iql_definition(cp,cfg));reset()
            for label,field,value in [('not_completed','status','running'),('completion_steps','steps',19999),('selection_split_violation','development_or_confirmation_used',True),('checkpoint_hash_mismatch','checkpoints',[dict(path=cp.name,step=20000,sha256='0'*64)])]:
                bad=deepcopy(completion);bad[field]=value;write(run/'completion.json',bad)
                rejected(label,lambda:ev.iql_definition(cp,cfg));reset()
            bad=deepcopy(provenance);bad['source_sha256'].pop(str(required[0].relative_to(fp)));write(run/'provenance.json',bad)
            rejected('missing_required_source_refused',lambda:ev.iql_definition(cp,cfg));reset()
            file=required[0];old=file.read_bytes();file.write_bytes(old+b'changed')
            rejected('source_drift_refused',lambda:ev.iql_definition(cp,cfg));file.write_bytes(old)
            bad=deepcopy(replay_data);bad['episodes']=279;write(replay/'manifest.json',bad)
            pbad=deepcopy(provenance);pbad['replay_manifest_sha256']=ev.sha(replay/'manifest.json');write(run/'provenance.json',pbad)
            rejected('replay_training_metadata_disagree',lambda:ev.iql_definition(cp,cfg));reset()
            # A completed four-update training smoke is separate from formal weights.
            scp=run/'smoke_step4.pt';scp.write_bytes(b'smoke fixture')
            pbad=deepcopy(provenance);pbad.update(smoke=True,planned_updates=4);write(run/'provenance.json',pbad)
            cbad=deepcopy(completion);cbad.update(status='smoke_completed',steps=4,sample_visits=4*256,final_checkpoint=None,checkpoints=[dict(path=scp.name,step=4,sha256=ev.sha(scp))]);write(run/'completion.json',cbad)
            rejected('smoke_weights_formal_eval_refused',lambda:ev.iql_definition(scp,cfg))
            check('explicit_smoke_allowed',ev.iql_definition(scp,cfg,True)[1]['smoke'])
            check('smoke_worker_allow_flag',ev.dependencies('iql_wide',scp,cfg,allow_smoke=True)[2][-1]=='--allow-smoke')
            reset()
            manifest=dict(checkpoint_sha256=ev.sha(cp),config_sha256=ev.sha(cfg),policy_training_provenance_sha256=ev.sha(run/'provenance.json'),iql_training_smoke=False)
            ready=dict(kind='IQL_wide',state_dim=1613,deterministic=True,checkpoint_sha256=manifest['checkpoint_sha256'],config_sha256=manifest['config_sha256'],provenance_sha256=manifest['policy_training_provenance_sha256'],smoke=False)
            ev.validate_iql_ready(ready,manifest);check('worker_ready_binding_valid',True)
            for field in ['kind','state_dim','deterministic','checkpoint_sha256','config_sha256','provenance_sha256','smoke']:
                bad=dict(ready);bad[field]='wrong';rejected('ready_'+field+'_mismatch',lambda bad=bad:ev.validate_iql_ready(bad,manifest))
            # Exercise actual failure serialization and original scoring on absent records.
            args=SimpleNamespace(kind='iql_wide',name='fixture_failure',checkpoint=cp,config=cfg,split='development',smoke=True,batch_size=2,method=None)
            with patch.object(ev,'JsonWorker',side_effect=RuntimeError('fixture GPU startup failure')):
                try:ev.run(args)
                except RuntimeError:pass
                else:raise AssertionError('Startup failure incorrectly succeeded')
            out=fr/'results/fixture_failure';summary=json.loads((out/'summary.json').read_text())
            check('technical_failure_retained_not_success',summary['status']=='technical_failure' and summary['count']==2)
            rows=[json.loads((out/item['raw_path']).read_text()) for item in summary['episodes']]
            check('unknown_tail_preserved_without_fake_BG',all(row['unknown_tail'] and not row['records'] and not row['raw_trajectory_available'] for row in rows))
            check('original_scorer_and_jobs_in_failed_summary',summary['provenance']['scorer_sha256']==protocol['scorer_sha256'] and summary['jobs_sha256']==ev.json_sha(ev.make_jobs(protocol,'development',True)))
            check('failed_run_has_complete_bound_manifest',json.loads((out/'manifest.json').read_text())['checkpoint_sha256']==ev.sha(cp))
            rejected('existing_output_never_overwritten',lambda:ev.run(args))
    check('existing34_32_58_checks_and_other_sources_unchanged',all(ev.sha(Path(p))==digest for p,digest in before.items()))
    check('no_Torch_or_simglucose_import',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    result=dict(status='passed',count=len(CHECKS),checks=CHECKS,fixture_only=True,checkpoint_deserialization=False,model_inference_executed=False,simulator_executed=False,remote_executed=False,evaluator_sha256=ev.sha(R/'evaluate_candidates.py'),test_source_sha256=ev.sha(Path(__file__).resolve()),original_checks_sha256={str(p.relative_to(R)):before[str(p)] for p in prior_checks},development_jobs_sha256=ev.json_sha(jobs),scorer_sha256=protocol['scorer_sha256'],limitation='Metadata and orchestration fixtures are not Torch inference, training, simulation efficacy or clinical evidence.')
    write(R/'checks/evaluate_iql_wide_mechanics.json',result)
    print(json.dumps(dict(status='passed',count=len(CHECKS),evaluator_sha256=result['evaluator_sha256'])))

if __name__=='__main__':main()
