"""Temporary fixture protocol only; never freeze the actual research selection."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(R))
import confirmation_gate as gate
import evaluate_candidates as ev
import physiologic_features as phys

checks=[]
def ok(name,value):
    assert value,name
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,FileExistsError,FileNotFoundError):checks.append(name)
    else:raise AssertionError('Invalid confirmation operation accepted: '+name)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))


def main():
    protected=list(R.glob('*.py'))+list((R/'configs').glob('*.json'))
    protected+=list((R/'checks').glob('*.json'))
    before={str(p):gate.sha(p) for p in protected}
    actual_confirmation_files=list((R/'confirmation').rglob('*')) if (R/'confirmation').exists() else []
    ast.parse((R/'confirmation_gate.py').read_text(),feature_version=(3,8))
    ok('python38_syntax',True)
    protocol=json.loads((R/'protocol.json').read_text())
    with tempfile.TemporaryDirectory(prefix='confirmation_fixture_',dir=R/'checks') as tmp:
        fp=Path(tmp);fr=fp/R.name;fr.mkdir()
        for actual in [R/n for n in ('protocol.json','confirmation_gate.py','evaluate_candidates.py','evaluate_confirmation.py',
                                     'summarize_panels.py','summarize_confirmation.py','build_report.py')]+[ev.SCORER,phys.NORMALIZER_PATH]:
            target=fp/actual.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(actual.read_bytes())
        runner=fr/'evaluate_confirmation.py'
        selected=fr/'selection_decision.md';selected.write_text('Fixture selection only, not a research candidate.\n')
        def binding(path):return dict(path=str(path.relative_to(fp)),sha256=gate.sha(path))
        def dependencies(kind,checkpoint,config,method=None,allow_smoke=False):
            assert not allow_smoke
            if kind not in ('hold','physiology'):raise ValueError('fixture supports fixed controllers only')
            return [fr/'evaluate_candidates.py',fp/protocol['scorer']],[fr/'protocol.json',fp/phys.NORMALIZER_PATH.relative_to(P)],None,None
        def method(mid,kind):
            out=fr/'results'/('development_'+mid);out.mkdir(parents=True)
            jobs=ev.make_jobs(protocol,'development')
            manifest=dict(kind=kind,method=None,split='development',smoke=False,jobs=jobs,jobs_sha256=ev.json_sha(jobs),checkpoint_sha256=None,config_sha256=None,scorer_sha256=protocol['scorer_sha256'])
            write(out/'manifest.json',manifest)
            episodes=[]
            for index,job in enumerate(jobs):
                raw=out/('raw%02d.json'%index);write(raw,dict(fixture=True,index=index))
                episodes.append(dict(key=ev.job_key(job),raw_path=raw.name,raw_sha256=gate.sha(raw)))
            summary=dict(status='completed',kind=kind,split='development',smoke=False,count=60,planned_count=60,jobs_sha256=ev.json_sha(jobs),provenance=dict(manifest_sha256=gate.sha(out/'manifest.json')),episodes=episodes)
            write(out/'summary.json',summary)
            sources,artifacts,_,_=dependencies(kind,None,None)
            return dict(id=mid,kind=kind,method=None,batch_size=12,checkpoint=None,config=None,worker=None,
                source_sha256=gate.hashes(sources+[fr/'confirmation_gate.py',runner,fr/'summarize_panels.py',
                                                 fr/'summarize_confirmation.py',fr/'build_report.py']),artifact_sha256=gate.hashes(artifacts),
                development_evidence=dict(manifest=binding(out/'manifest.json'),summary=binding(out/'summary.json')))
        with patch.multiple(gate,R=fr,P=fp,__file__=str(fr/'confirmation_gate.py')),patch.object(phys,'NORMALIZER_PATH',fp/phys.NORMALIZER_PATH.relative_to(P)),patch.object(ev,'dependencies',side_effect=dependencies):
            # The fixture dependency hook above needs the original relative normalizer.
            normalizer=fp/'Loop数据集/训练管线_v2/prepared/normalization.json'
            def fixture_dependencies(kind,checkpoint,config,method=None,allow_smoke=False):
                assert not allow_smoke
                if kind not in ('hold','physiology'):raise ValueError('fixture fixed controllers only')
                return [fr/'evaluate_candidates.py',fp/protocol['scorer']],[fr/'protocol.json',normalizer],None,None
            with patch.object(ev,'dependencies',side_effect=fixture_dependencies):
                # Method construction uses this helper's equivalent original paths.
                dependencies=fixture_dependencies
                plan=dict(schema=1,selection_finalized=True,confirmation_previously_generated=False,smoke=False,split='confirmation',
                    required_comparator_ids=['hold_reference'],selected_internal_ids=['fixture_internal'],
                    protocol=binding(fr/'protocol.json'),scorer=binding(fp/protocol['scorer']),normalizer=binding(normalizer),
                    confirmation_runner=binding(runner),selection_evidence=binding(selected),
                    methods=[method('hold_reference','hold'),method('fixture_internal','physiology')])
                path=fr/'fixture_plan.json';write(path,plan)
                audit=gate.validate_plan(plan)
                ok('read_only_validation_has60_and_no_freeze_or_jobs',audit['case_count']==60 and not (fr/'confirmation').exists())
                for field,value,label in [('selection_finalized',False,'unfinished_selection'),('confirmation_previously_generated',True,'prior_exposure_attestation'),('smoke',True,'smoke_confirmation'),('split','development','wrong_split'),('selected_internal_ids',[],'missing_internal_selection'),('required_comparator_ids',[],'missing_comparators')]:
                    bad=deepcopy(plan);bad[field]=value;reject(label,lambda bad=bad:gate.validate_plan(bad))
                bad=deepcopy(plan);bad['methods'][1]['id']='hold_reference';reject('duplicate_methods',lambda:gate.validate_plan(bad))
                bad=deepcopy(plan);bad['methods'][1]['source_sha256'].pop(next(iter(bad['methods'][1]['source_sha256'])));reject('source_manifest_not_exact',lambda:gate.validate_plan(bad))
                bad=deepcopy(plan);bad['methods'][1]['batch_size']=0;reject('invalid_fixed_batch_size',lambda:gate.validate_plan(bad))
                bad=deepcopy(plan);bad['confirmation_runner']=binding(fr/'evaluate_candidates.py');reject('existing_development_evaluator_not_confirmation_runner',lambda:gate.validate_plan(bad))
                bad=deepcopy(plan);bad['normalizer']['sha256']='0'*64;reject('normalizer_hash_drift',lambda:gate.validate_plan(bad))
                raw=fr/'results/development_hold_reference/raw00.json';original=raw.read_bytes();raw.write_text('changed')
                reject('development_raw_evidence_drift',lambda:gate.validate_plan(plan));raw.write_bytes(original)
                exposed=fr/'results/prior_confirmation/manifest.json';write(exposed,dict(split='confirmation'))
                reject('retained_confirmation_metadata_blocks_first_freeze',lambda:gate.validate_plan(plan));exposed.unlink();exposed.parent.rmdir()
                conflict=deepcopy(protocol);conflict['development_scenario_seeds']=[103901,103102]
                reject('reserved_seed_overlap',lambda:gate.case_specs(conflict))
                frozen=gate.freeze(path)
                ok('fixture_freeze_pins_explicit_set_and_seal',gate.load_freeze(frozen)['plan']==plan and (frozen.parent/'seal.json').exists())
                reject('second_freeze_or_selection_replacement_forbidden',lambda:gate.freeze(path))
                reject('unknown_method_cannot_authorize',lambda:gate.authorize(frozen,'unlisted'))
                ticket_path=gate.authorize(frozen,'hold_reference');ticket=gate.validate_ticket(ticket_path)
                ok('single_ticket_has_full_fixed60_jobs',len(ticket['jobs'])==60 and {j['seed'] for j in ticket['jobs']}=={103901,103902} and ticket['smoke'] is False)
                reject('duplicate_or_partial_rerun_forbidden',lambda:gate.authorize(frozen,'hold_reference'))
                original_ticket=ticket_path.read_bytes();bad=deepcopy(ticket);bad['jobs']=bad['jobs'][:2];write(ticket_path,bad)
                reject('ticket_subset_tampering_rejected',lambda:gate.validate_ticket(ticket_path));ticket_path.write_bytes(original_ticket)
                partial=ticket_path.parent/'first_observed_failure.json';write(partial,dict(fixture=True,preserved=True))
                failure=gate.record_result(ticket_path,failure_reason='fixture infrastructure failure before all cases')
                ok('failure_retains_existing_artifacts_and_unknown60',failure['status']=='technical_failure' and failure['unknown_case_count']==60 and str(partial.relative_to(fp)) in failure['result_sha256'])
                reject('failure_cannot_be_silently_retried',lambda:gate.authorize(frozen,'hold_reference'))
                reject('finish_cannot_overwrite_prior_failure',lambda:gate.record_result(ticket_path,failure_reason='retry'))
                second=gate.authorize(frozen,'fixture_internal');second_ticket=gate.validate_ticket(second)
                ok('remaining_frozen_method_can_continue_after_failure',len(second_ticket['jobs'])==60 and second_ticket['jobs_sha256']==ticket['jobs_sha256'])
                gate.start_run(second)
                reject('ticket_execution_start_is_single_use',lambda:gate.start_run(second))
                # A fully reported fixture panel may contain native terminations;
                # coverage stays in the original scorer rather than becoming success.
                out=second.parent;episodes=[];m=second_ticket['method']
                provenance=dict(freeze_sha256=second_ticket['freeze_sha256'],ticket_sha256=gate.sha(second),scorer_sha256=second_ticket['scorer']['sha256'],protocol_sha256=second_ticket['protocol']['sha256'],normalization_sha256=second_ticket['normalizer']['sha256'],worker_sha256=None,config_sha256=None,checkpoint_sha256=None,source_sha256=m['source_sha256'])
                manifest=dict(schema=1,kind=m['kind'],method=None,protocol=protocol,split='confirmation',smoke=False,
                    confirmation=True,tuning_split=False,jobs=second_ticket['jobs'],jobs_sha256=second_ticket['jobs_sha256'],
                    artifact_sha256=m['artifact_sha256'],**provenance)
                write(out/'manifest.json',manifest)
                provenance.update(manifest_sha256=gate.sha(out/'manifest.json'),jobs_sha256=second_ticket['jobs_sha256'])
                for index,job in enumerate(second_ticket['jobs']):
                    records=[dict(minute=5*i,cgm_mg_dl=120.,bg_mg_dl=120.,requested_basal_u_h=1.,delivered_basal_u_h=1.,bolus_u=0.,meal_g=0.,warmup=i<=72) for i in range(1,74)]
                    metrics=ev.summarize(records,792,True)
                    raw=dict(job=job,records=records,failure_reason='native_environment_done',raw_trajectory_available=True,unknown_tail=True,metrics=metrics,evaluation_provenance=provenance)
                    rp=out/('raw%02d.json'%index);write(rp,raw)
                    episodes.append(dict(key=ev.job_key(job),raw_path=rp.name,raw_sha256=gate.sha(rp),metrics=metrics,technical_failure_reason=None,failure_reason='native_environment_done'))
                summary=dict(status='completed',kind=m['kind'],method=None,split='confirmation',smoke=False,count=60,planned_count=60,jobs_sha256=second_ticket['jobs_sha256'],provenance=provenance,episodes=episodes)
                summary_path=out/'summary.json';write(summary_path,summary)
                ok('full60_original_scorer_result_validates',gate.check_result(second_ticket,summary_path)[0]=='completed')
                bad=deepcopy(summary);bad['count']=2;bad['episodes']=bad['episodes'][:2];write(summary_path,bad)
                reject('partial_confirmation_cannot_be_success',lambda:gate.check_result(second_ticket,summary_path));write(summary_path,summary)
                bad=deepcopy(summary);bad['provenance']['config_sha256']='0'*64;write(summary_path,bad)
                reject('changed_model_config_binding_rejected',lambda:gate.check_result(second_ticket,summary_path));write(summary_path,summary)
                bad=deepcopy(summary);bad['episodes'][0]['metrics']['bg']['coverage_pct']=100.;write(summary_path,bad)
                reject('unknown_tail_cannot_be_relabelled_complete',lambda:gate.check_result(second_ticket,summary_path));write(summary_path,summary)
                old=runner.read_bytes();runner.write_bytes(old+b'# changed after authorization\n')
                reject('source_change_after_partial_panel_rejected',lambda:gate.validate_ticket(second))
                final=gate.record_result(second,summary_path=summary_path)
                ok('dependency_drift_is_preserved_as_failed_finish',final['status']=='technical_failure' and 'validation failed' in final['error'])
                runner.write_bytes(old)
                reject('invalid_result_cannot_be_retried',lambda:gate.authorize(frozen,'fixture_internal'))
                original=frozen.read_bytes();changed=gate.read(frozen);changed['plan']['selected_internal_ids']=['replacement'];write(frozen,changed)
                reject('selection_cannot_change_after_first_confirmation',lambda:gate.load_freeze(frozen));frozen.write_bytes(original)
                # Full40 completion gate is tested separately with controlled update binding.
                train=fr/'results/PPO_fixture';train.mkdir();cfg=train/'config.json';write(cfg,dict(iterations=40))
                write(train/'completion.json',dict(status='completed',iterations=16))
                reject('PPO08_selection_waits_for_full40_training',lambda:gate.training_completion('ppo',train/'policy_iter08.pt',cfg))
                write(train/'completion.json',dict(status='completed',iterations=40))
                with patch.object(ev,'policy_evaluation_binding',return_value=({},[],[],b'')) as bound:
                    extra=gate.training_completion('ppo_mean',train/'policy_iter08.pt',cfg)
                    ok('completed40_allows_preselected08_and_binds40',train/'policy_iter40.pt' in extra and bound.call_args.args[1].name=='policy_iter40.pt')
    ok('actual_research_confirmation_never_created',actual_confirmation_files==(list((R/'confirmation').rglob('*')) if (R/'confirmation').exists() else []))
    ok('existing_evaluator_models_workers_configs_unchanged',all(gate.sha(Path(p))==digest for p,digest in before.items()))
    ok('no_Torch_or_simulator_loaded',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    result=dict(status='passed',count=len(checks),checks=checks,fixture_only=True,actual_selection_frozen=False,actual_confirmation_jobs_generated=False,remote_executed=False,gate_sha256=gate.sha(R/'confirmation_gate.py'),evaluator_sha256=gate.sha(R/'evaluate_candidates.py'),test_source_sha256=gate.sha(Path(__file__).resolve()),limitation='Filesystem state-machine and metadata fixtures only. A reviewed confirmation runner is not implemented or connected; no final research method selection is made.')
    write(R/'checks/confirmation_gate_integrated_mechanics.json',result)
    print(json.dumps(dict(status='passed',count=len(checks),gate_sha256=result['gate_sha256'])))

if __name__=='__main__':main()
