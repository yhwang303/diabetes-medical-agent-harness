"""Temporary metadata/transport fixtures; never executes a real confirmation run."""
import ast
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(R))
import confirmation_gate as gate
import evaluate_candidates as ev
import physiologic_features as phys
import summarize_panels as agg
import summarize_confirmation as adapter
spec=importlib.util.spec_from_file_location('confirmation_report_fixture',R/'build_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

checks=[]
def ok(name,value):
    assert value,name
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,FileExistsError,FileNotFoundError,KeyError,RuntimeError):checks.append(name)
    else:raise AssertionError('Accepted invalid fixture: '+name)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    protected=list(R.glob('*.py'))+list((R/'configs').glob('*.json'))+list((R/'checks').glob('*.json'))
    protected+=[ev.SCORER]
    before={str(p):gate.sha(p) for p in protected}
    real_confirmation=list((R/'confirmation').rglob('*')) if (R/'confirmation').exists() else []
    protocol=json.loads((R/'protocol.json').read_text())
    for name in ('confirmation_gate.py','evaluate_candidates.py','evaluate_confirmation.py','summarize_confirmation.py','build_report.py'):
        ast.parse((R/name).read_text(),feature_version=(3,8));ok('syntax38_'+name,True)
    reject('ordinary_make_jobs_still_rejects_confirmation',lambda:ev.make_jobs(protocol,'confirmation'))
    mean,scale,_=phys.normalization();history=ev.np.zeros((72,22),dtype=ev.np.float32)
    history[:,1]=(1/12-mean[1])/scale[1];history[:,6]=1

    # Fake transport emits declared fixtures only, with the actual runner loop.
    class Pipe:
        def __init__(self):self.job=None;self.first=True;self.abort=None;self.action=None
        def poll(self,timeout):return True
        def recv(self):
            if self.first:
                self.first=False;return dict(done=False,history=history.tolist())
            stop=72 if self.abort else 864
            records=[dict(minute=5*i,warmup=i<=72,cgm_mg_dl=120.,bg_mg_dl=120.,
                requested_basal_u_h=1. if self.action is None else self.action,
                delivered_basal_u_h=1.,bolus_u=0.,meal_g=0.) for i in range(1,stop+1)]
            return dict(done=True,result=dict(job=self.job,records=records,failure_reason=self.abort,
                                              unknown_tail=bool(self.abort),fixture_only=True))
        def send(self,value):
            if 'abort' in value:self.abort=value['abort']
            else:self.action=value['action_u_h']
        def close(self):pass
    class Process:
        def __init__(self,child,job):child.parent.job=job;self.alive=False
        def start(self):self.alive=True
        def join(self,timeout):self.alive=False
        def is_alive(self):return self.alive
        def terminate(self):self.alive=False
    class Context:
        def Pipe(self):
            pipe=Pipe();return pipe,SimpleNamespace(parent=pipe,close=lambda:None)
        def Process(self,target,args):return Process(*args)

    with tempfile.TemporaryDirectory(prefix='confirmation_integration_fixture_',dir=R/'checks') as tmp:
        fp=Path(tmp);fr=fp/R.name;fr.mkdir();(fr/'checks').mkdir()
        names=('protocol.json','confirmation_gate.py','evaluate_candidates.py','evaluate_confirmation.py',
               'summarize_panels.py','summarize_confirmation.py','build_report.py')
        for actual in [R/n for n in names]+[ev.SCORER,phys.NORMALIZER_PATH]:
            target=fp/actual.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(actual.read_bytes())
        scorer=fp/ev.SCORER.relative_to(P);normalizer=fp/phys.NORMALIZER_PATH.relative_to(P)
        selected=fr/'fixture_selection.md';selected.write_text('Fixture only, no research method selection.')
        checkpoint=fr/'fixture_legacy.pt';checkpoint.write_bytes(b'fixture-not-a-model')
        worker=fr/'fixture_worker.py';worker.write_text('# fixture transport only; never executed\n')
        write(fr/'checks/frozen_selected_version.json',dict(fixture_only=True))
        source=[fr/'evaluate_candidates.py',scorer]
        def binding(path):return dict(path=str(path.relative_to(fp)),sha256=gate.sha(path))
        def dependencies(kind,checkpoint,config,method=None,allow_smoke=False):
            assert kind in ('hold','legacy') and checkpoint is None and config is None and method is None and not allow_smoke
            if kind=='legacy':
                return source+[worker],[fr/'protocol.json',normalizer,fr/'fixture_legacy.pt'],['fixture-python','-u',str(worker)],fr/'fixture_legacy.pt'
            return source,[fr/'protocol.json',normalizer],None,None
        def method(mid):
            kind='legacy' if mid=='cleanup_failure' else 'hold'
            folder=fr/'results'/('development_'+mid);folder.mkdir(parents=True)
            jobs=ev.make_jobs(protocol,'development')
            manifest=dict(kind=kind,method=None,split='development',smoke=False,jobs=jobs,jobs_sha256=ev.json_sha(jobs),
                          checkpoint_sha256=gate.sha(checkpoint) if kind=='legacy' else None,config_sha256=None,scorer_sha256=protocol['scorer_sha256'])
            write(folder/'manifest.json',manifest);episodes=[]
            for index,job in enumerate(jobs):
                path=folder/('fixture%02d.json'%index);write(path,dict(fixture_only=True,index=index))
                episodes.append(dict(key=ev.job_key(job),raw_path=path.name,raw_sha256=gate.sha(path)))
            write(folder/'summary.json',dict(status='completed',kind=kind,split='development',smoke=False,count=60,planned_count=60,
                jobs_sha256=ev.json_sha(jobs),provenance=dict(manifest_sha256=gate.sha(folder/'manifest.json')),episodes=episodes))
            src,art,_,_=dependencies(kind,None,None)
            return dict(id=mid,kind=kind,method=None,batch_size=12,checkpoint=binding(checkpoint) if kind=='legacy' else None,
                config=None,worker=binding(worker) if kind=='legacy' else None,
                source_sha256=gate.hashes(src+[fr/n for n in ('confirmation_gate.py','evaluate_confirmation.py',
                    'summarize_panels.py','summarize_confirmation.py','build_report.py')]),
                artifact_sha256=gate.hashes(art+([fr/'checks/frozen_selected_version.json'] if kind=='legacy' else [])),
                development_evidence=dict(manifest=binding(folder/'manifest.json'),summary=binding(folder/'summary.json')))
        with ExitStack() as stack:
            for module in (gate,ev,agg,adapter):stack.enter_context(patch.multiple(module,R=fr,P=fp))
            stack.enter_context(patch.object(gate,'__file__',str(fr/'confirmation_gate.py')))
            stack.enter_context(patch.object(adapter,'__file__',str(fr/'summarize_confirmation.py')))
            stack.enter_context(patch.object(ev,'COMMON_SOURCES',source))
            stack.enter_context(patch.object(ev,'SCORER',scorer));stack.enter_context(patch.object(agg,'SCORER',scorer))
            stack.enter_context(patch.object(ev,'NORMALIZER_PATH',normalizer));stack.enter_context(patch.object(phys,'NORMALIZER_PATH',normalizer))
            stack.enter_context(patch.object(ev,'dependencies',side_effect=dependencies))
            stack.enter_context(patch.object(ev.mp,'get_context',return_value=Context()))
            class CleanupWorker:
                def __init__(self,*args):self.process=SimpleNamespace(returncode=0,poll=lambda:0)
                def receive(self):return dict(ready=True,fixture_only=True)
                def request(self,payload):return dict(actions_u_h=payload['anchor_u_h'])
                def close(self):raise RuntimeError('fixture cleanup failure')
            stack.enter_context(patch.object(ev,'JsonWorker',CleanupWorker))
            stack.enter_context(redirect_stdout(io.StringIO()))
            plan=dict(schema=1,selection_finalized=True,confirmation_previously_generated=False,smoke=False,split='confirmation',
                required_comparator_ids=['reference'],selected_internal_ids=['startup_failure','loop_failure','cleanup_failure'],
                protocol=binding(fr/'protocol.json'),scorer=binding(scorer),normalizer=binding(normalizer),
                confirmation_runner=binding(fr/'evaluate_confirmation.py'),selection_evidence=binding(selected),
                methods=[method(mid) for mid in ('reference','startup_failure','loop_failure','cleanup_failure')])
            plan_path=fr/'fixture_plan.json';write(plan_path,plan)
            gate.validate_plan(plan)
            ok('validation_does_not_generate_confirmation_jobs',not (fr/'confirmation').exists())
            frozen=gate.freeze(plan_path)
            ok('freeze_does_not_generate_confirmation_tickets',not list((frozen.parent/'runs').iterdir()))
            ticket=gate.authorize(frozen,'reference');ticket_data=gate.validate_ticket(ticket)
            ok('authorize_has_fixed60_after_freeze',len(ticket_data['jobs'])==60 and {j['seed'] for j in ticket_data['jobs']}==set(protocol['confirmation_scenario_seeds']))
            stray=ticket.parent/'stray.json';write(stray,dict(fixture_only=True))
            reject('preexisting_output_blocks_start',lambda:ev.run_confirmation(ticket));stray.unlink()
            out=ev.run_confirmation(ticket)
            summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
            ok('shared_loop_writes_complete_fixed60_fixture',summary['status']=='completed' and summary['count']==60 and
               all(e['metrics']['bg']['coverage_pct']==100 for e in summary['episodes']))
            ok('confirmation_manifest_flags_and_sources',manifest['confirmation'] is True and manifest['tuning_split'] is False and
               manifest['source_sha256']==plan['methods'][0]['source_sha256'])
            reject('same_finished_ticket_cannot_execute_twice',lambda:ev.run_confirmation(ticket))
            ok('finished_ticket_remains_readable_for_audit',gate.validate_ticket(ticket)['method_id']=='reference')
            started=json.loads((out/'start.json').read_text())
            ok('exclusive_start_bound_to_ticket',started['ticket_sha256']==gate.sha(ticket))
            original_manifest=(out/'manifest.json').read_bytes();(out/'manifest.json').unlink()
            reject('missing_manifest_rejected',lambda:gate.check_result(ticket_data,out/'summary.json'))
            (out/'manifest.json').write_bytes(original_manifest)
            raw_path=out/summary['episodes'][0]['raw_path'];original_raw=raw_path.read_bytes();original_summary=(out/'summary.json').read_bytes()
            raw=json.loads(original_raw);raw['records'][0]['minute']=365
            write(raw_path,raw);summary['episodes'][0]['raw_sha256']=gate.sha(raw_path);write(out/'summary.json',summary)
            reject('shifted_minute_rejected_despite_rehashed_raw',lambda:gate.check_result(ticket_data,out/'summary.json'))
            raw_path.write_bytes(original_raw);(out/'summary.json').write_bytes(original_summary)
            raw=json.loads(original_raw);raw['records'][0]['warmup']=False;raw['records'][-1]['warmup']=True
            raw['metrics']=ev.summarize(raw['records'],792,False)
            write(raw_path,raw);summary=json.loads(original_summary);summary['episodes'][0].update(raw_sha256=gate.sha(raw_path),metrics=raw['metrics']);write(out/'summary.json',summary)
            reject('wrong_warmup_rejected_even_with_original_rescore',lambda:gate.check_result(ticket_data,out/'summary.json'))
            raw_path.write_bytes(original_raw);(out/'summary.json').write_bytes(original_summary)
            raw=json.loads(original_raw);raw['evaluation_provenance']['freeze_sha256']='0'*64
            write(raw_path,raw);summary=json.loads(original_summary);summary['episodes'][0]['raw_sha256']=gate.sha(raw_path);write(out/'summary.json',summary)
            reject('raw_frozen_provenance_drift_rejected',lambda:gate.check_result(ticket_data,out/'summary.json'))
            raw_path.write_bytes(original_raw);(out/'summary.json').write_bytes(original_summary)
            for mid in ('startup_failure','loop_failure'):
                tp=gate.authorize(frozen,mid)
                reject('adapter_rejects_unfinished_frozen_set',lambda:adapter.build(frozen,'reference',Path('checks/panels_confirmation_premature.json')))
                if mid=='startup_failure':context=patch.object(ev,'make_controller',side_effect=RuntimeError('fixture startup failure'))
                else:context=patch.object(ev,'make_controller',return_value=SimpleNamespace(action=lambda h,a:(_ for _ in ()).throw(RuntimeError('fixture action failure'))))
                with context:reject(mid+'_raises_and_preserves_evidence',lambda tp=tp:ev.run_confirmation(tp))
                failed=json.loads((tp.parent/'summary.json').read_text())
                ok(mid+'_retains_all60_raw',failed['status']=='technical_failure' and failed['count']==60 and
                   len(list((tp.parent/'trajectories').glob('*.json')))==60 and all(e['metrics']['failed'] for e in failed['episodes']))
                ok(mid+'_ended_without_retry',gate.events(frozen.parent)[-1]['event']=='finished' and
                   gate.events(frozen.parent)[-1]['complete_fixed_case_reporting'] is True)
                reject(mid+'_ticket_cannot_restart',lambda tp=tp:ev.run_confirmation(tp))
            tp=gate.authorize(frozen,'cleanup_failure')
            reject('cleanup_failure_raises_after_retaining_all_raw',lambda:ev.run_confirmation(tp))
            failed=json.loads((tp.parent/'summary.json').read_text())
            ok('cleanup_failure_retains60_full_raw_without_fabricating_patient_failure',failed['status']=='technical_failure' and
               failed['count']==60 and len(list((tp.parent/'trajectories').glob('*.json')))==60 and
               all(not e['metrics']['failed'] and e['metrics']['bg']['coverage_pct']==100 for e in failed['episodes']))
            reject('cleanup_failure_ticket_cannot_restart',lambda:ev.run_confirmation(tp))
            payload_path=adapter.build(frozen,'reference',Path('checks/panels_confirmation_fixture.json'))
            payload=json.loads(payload_path.read_text());adapter.validate_bundle(payload)
            ok('adapter_really_rescores_all240_raw',sum(p['episodes'] for p in payload['panels'].values())==240 and
               all(p['scoring_verified_exact'] for p in payload['panels'].values()))
            ok('technical_failures_remain_observed',all(payload['panels']['confirmation_'+mid]['complete'] is None
               for mid in ('startup_failure','loop_failure','cleanup_failure')))
            reject('adapter_never_overwrites_output',lambda:adapter.build(frozen,'reference',Path('checks/panels_confirmation_fixture.json')))
            bad=deepcopy(payload);bad['panels'].pop('confirmation_loop_failure')
            reject('frozen_set_subset_rejected',lambda:adapter.validate_bundle(bad))
            bad=deepcopy(payload);bad['confirmation_audit']['metadata'][bad['confirmation_audit']['freeze_path']]['text']+=' '
            reject('metadata_byte_hash_drift_rejected',lambda:adapter.validate_bundle(bad))
            verified=report.load_panels(payload_path,protocol,confirmation=True)
            ok('report_accepts_real_adapter_audit_with_failed_rows',len(verified['panels'])==4)
            text=dict(schema=1,paragraphs=['Fixture only; no real confirmation.'],panels={n:dict(method_id='d06_frozen' if p['controller']=='legacy' else 'hold',label=n,group='comparison') for n,p in payload['panels'].items()},
                confirmation_review=dict(reviewed=True,reviewer='fixture only',panel_manifest_sha256={n:p['manifest_sha256'] for n,p in payload['panels'].items()}))
            registry=json.loads((R/'method_registry.json').read_text())
            registry=deepcopy(registry)
            next(m for m in registry['methods'] if m['id']=='d06_frozen')['checkpoint']['sha256']=gate.sha(checkpoint)
            report.validate_summary(text,payload['panels'],registry,'final',[payload_path])
            rows=[];cells={};report.score_table(list(payload['panels'].values()),text['panels'],'bg',rows,cells)
            ok('failed_confirmation_never_bolded',not any(v['best'] and v['panel']!='confirmation_reference' for v in cells.values()))
            # Ordinary run uses the same mocked transport and keeps its old identity/paths.
            args=SimpleNamespace(kind='hold',method=None,name='development_regression',split='development',smoke=False,
                                 checkpoint=None,config=None,batch_size=12)
            development=ev.run(args);d=json.loads((development/'manifest.json').read_text())
            ok('ordinary_development_jobs_flags_and_output_unchanged',development==fr/'results/development_regression' and
               d['jobs']==ev.make_jobs(protocol,'development') and d['confirmation'] is False and d['tuning_split'] is True and
               'freeze_sha256' not in d)
            agg.load_panel(development,'development_regression');ok('ordinary_raw_format_still_original_scorer_exact',True)
            reject('ordinary_output_still_exclusive',lambda:ev.run(args))
            args.split='confirmation';args.name='forbidden_ordinary';reject('ordinary_run_still_cannot_generate_confirm',lambda:ev.run(args))
            ok('ordinary_rejection_created_no_directory',not (fr/'results/forbidden_ordinary').exists())
    # Current development JSON remains readable after adding a separate adapter.
    report.load_panels(R/'checks/panels_development_12_methods_r1.json',protocol)
    ok('existing12_development_panel_audit_still_valid',True)
    ok('all_existing_sources_configs_checks_unchanged_during_fixture',all(gate.sha(Path(p))==v for p,v in before.items()))
    ok('real_confirmation_directory_never_created',real_confirmation==(list((R/'confirmation').rglob('*')) if (R/'confirmation').exists() else []))
    ok('no_Torch_or_simulator_loaded',not any(n=='torch' or n.startswith('simglucose') for n in sys.modules))
    result=dict(status='passed',count=len(checks),checks=checks,fixture_only=True,actual_selection_frozen=False,
        actual_confirmation_jobs_generated=False,remote_executed=False,training_or_simulation_executed=False,
        source_sha256={n:gate.sha(R/n) for n in ('confirmation_gate.py','evaluate_candidates.py','evaluate_confirmation.py','summarize_confirmation.py','build_report.py')},
        original_aggregation_sha256=gate.sha(R/'summarize_panels.py'),original_scorer_sha256=gate.sha(ev.SCORER),test_source_sha256=gate.sha(Path(__file__)))
    write(R/'checks/confirmation_integration_mechanics.json',result)
    print(json.dumps(dict(status='passed',count=len(checks))))


if __name__=='__main__':main()
