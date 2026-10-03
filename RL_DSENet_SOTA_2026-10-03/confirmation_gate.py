"""Independent confirmation authorization and single-use execution identity.

No final methods are inferred. Validation is read-only. A caller must explicitly
freeze a completed development selection before authorize can generate jobs.
No simulation, training, remote access or automatic retry is implemented here.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

R=Path(__file__).resolve().parent
P=R.parent
KINDS={'hold','physiology','legacy','retained','ppo','world','world_ppo','iql_wide','ppo_mean','world_ppo_mean'}
RESERVED=[103901,103902]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def project_path(relative):
    if not isinstance(relative,str) or Path(relative).is_absolute():
        raise ValueError('Bindings use project-relative paths')
    path=(P/relative).resolve();path.relative_to(P.resolve())
    return path


def file_binding(binding):
    if set(binding)!={'path','sha256'}:
        raise ValueError('A file binding needs exactly path and sha256')
    path=project_path(binding['path'])
    if sha(path)!=binding['sha256']:
        raise ValueError('File hash changed: '+binding['path'])
    return path


def hashes(paths):
    return {str(p.resolve().relative_to(P.resolve())):sha(p) for p in sorted(set(paths))}


def case_specs(protocol):
    if (protocol['confirmation_scenario_seeds']!=RESERVED or protocol['patients']!=list(range(1,11))
            or protocol['bolus_factors']!=[.8,1.,1.2] or protocol['warmup_minutes']!=360
            or protocol['total_minutes']!=4320 or protocol['training_seed']!=260915):
        raise ValueError('The fixed 60-case confirmation contract changed')
    for key in ('train_scenario_seeds','world_validation_scenario_seeds','development_scenario_seeds','smoke_scenario_seeds'):
        if set(protocol[key])&set(RESERVED):raise ValueError('Confirmation seed reused in another split')
    low,high=protocol['ppo_train_scenario_seed_range_inclusive']
    if set(range(low,high+1))&set(RESERVED):raise ValueError('Confirmation seed reused for PPO training')
    return [dict(patient=p,seed=s,group='bolus_%.1f'%f,bolus_factor=f,total_minutes=4320)
            for p in protocol['patients'] for f in protocol['bolus_factors'] for s in RESERVED]


def prior_exposure_audit():
    """Audit retained metadata; cannot prove absence of unlogged external runs."""
    inspected={}
    for root in (R/'results',R/'data',R/'cache',R/'confirmation'):
        if not root.exists():continue
        for path in root.rglob('*.json'):
            if path.name not in ('manifest.json','summary.json','config.json','ticket.json') and not any('s%d'%s in path.name for s in RESERVED):continue
            value=read(path);inspected[str(path.relative_to(P))]=sha(path)
            if not isinstance(value,dict):continue
            config=value.get('config',{})
            jobs=value.get('jobs',[])
            if value.get('job'):jobs=jobs+[value['job']]
            exposed=(value.get('split')=='confirmation' or config.get('split')=='confirmation'
                or any(job.get('seed') in RESERVED for job in jobs)
                or value.get('base_scenario_seed') in RESERVED or config.get('base_scenario_seed') in RESERVED)
            if exposed:raise ValueError('Confirmation evidence already exists before freeze: '+str(path))
    return dict(scope='retained research results/data/cache/confirmation metadata and reserved-seed named trajectories',
                independently_proves_no_unlogged_external_runs=False,metadata_sha256=inspected)


def training_completion(kind,checkpoint,config):
    """Require completed training even when selecting an earlier PPO update."""
    family={'ppo_mean':'ppo','world_ppo_mean':'world_ppo'}.get(kind,kind)
    extra=[]
    if family in ('ppo','world_ppo'):
        run=config.parent;c=read(config);path=run/'completion.json';completed=read(path)
        expected='completed' if family=='ppo' else 'completed_candidate_only'
        if completed['status']!=expected or completed['iterations']!=40 or c['iterations']!=40:
            raise ValueError('All 40 PPO iterations must finish before confirmation freeze')
        if family=='world_ppo' and completed['run_mode']!='formal':raise ValueError('Smoke PPO is not confirmation eligible')
        # The finished last update is bound too; the selected update may be 08/16/32.
        import evaluate_candidates as evaluator
        evaluator.policy_evaluation_binding(family,run/'policy_iter40.pt',config,False)
        extra=[path,run/'policy_iter40.pt',run/'history.jsonl']
    elif family=='world':
        # dependencies already requires completed4000/configured4000/no override.
        extra=[]
    elif family=='iql_wide':
        # dependencies already requires completed final20000, with no smoke.
        extra=[]
    elif family in ('legacy','retained'):
        # These are immutable historical selections, not newly running trainers.
        extra=[R/'checks/frozen_selected_version.json'] if family=='legacy' else [R/'configs/retained_methods.json']
    return extra


def method_dependencies(method,runner):
    import evaluate_candidates as evaluator
    kind=method['kind']
    if kind not in KINDS or type(method['batch_size']) is not int or not 1<=method['batch_size']<=60:
        raise ValueError('Unknown method kind or invalid fixed batch size')
    checkpoint=file_binding(method['checkpoint']) if method['checkpoint'] is not None else None
    config=file_binding(method['config']) if method['config'] is not None else None
    supplied=kind in ('ppo','world','world_ppo','iql_wide','ppo_mean','world_ppo_mean')
    if supplied!=(config is not None) or (kind in ('hold','physiology'))!=(checkpoint is None):
        raise ValueError('Method needs explicit checkpoint/config bindings appropriate to its kind')
    if (kind=='retained')!=(method['method'] is not None):raise ValueError('Only retained uses a method key')
    sources,artifacts,command,actual=evaluator.dependencies(kind,checkpoint if supplied else None,config,method=method['method'],allow_smoke=False)
    if actual!=checkpoint:raise ValueError('Selected checkpoint differs from the actual worker dependency')
    worker=file_binding(method['worker']) if method['worker'] is not None else None
    if (None if command is None else Path(command[2]).resolve())!=worker:
        raise ValueError('Explicit worker differs from the evaluator route')
    sources=sources+[Path(__file__).resolve(),runner,R/'summarize_panels.py',
                     R/'summarize_confirmation.py',R/'build_report.py']
    artifacts=artifacts+training_completion(kind,checkpoint,config)
    if hashes(sources)!=method['source_sha256'] or hashes(artifacts)!=method['artifact_sha256']:
        raise ValueError('Method source/dependency manifest is not exact: '+method['id'])
    return command


def method_config_sha(method):
    if method['config'] is not None:return method['config']['sha256']
    if method['kind']=='retained':return method['artifact_sha256'][str((R/'configs/retained_methods.json').relative_to(P))]
    return None


def development_evidence(method):
    import evaluate_candidates as evaluator
    evidence=method['development_evidence']
    manifest_path=file_binding(evidence['manifest']);summary_path=file_binding(evidence['summary'])
    manifest=read(manifest_path);summary=read(summary_path)
    protocol=read(R/'protocol.json');jobs=evaluator.make_jobs(protocol,'development')
    if (manifest_path.parent!=summary_path.parent or manifest['kind']!=method['kind']
            or manifest.get('method')!=method['method'] or manifest['split']!='development'
            or manifest['smoke'] is not False or summary['status']!='completed'
            or summary['count']!=60 or summary['planned_count']!=60 or summary['split']!='development'
            or summary['smoke'] is not False or summary['jobs_sha256']!=manifest['jobs_sha256']
            or manifest['jobs']!=jobs or manifest['jobs_sha256']!=evaluator.json_sha(jobs)
            or manifest['scorer_sha256']!=protocol['scorer_sha256']
            or summary['provenance']['manifest_sha256']!=sha(manifest_path)
            or manifest['checkpoint_sha256']!=(method['checkpoint']['sha256'] if method['checkpoint'] else None)
            or manifest.get('config_sha256')!=method_config_sha(method)):
        raise ValueError('Method lacks its exact completed60 development evidence: '+method['id'])
    for item in summary['episodes']:
        raw=(summary_path.parent/item['raw_path']).resolve();raw.relative_to(summary_path.parent.resolve())
        if sha(raw)!=item['raw_sha256']:raise ValueError('Development raw evidence changed')
    keys=[item['key'] for item in summary['episodes']]
    if len(keys)!=60 or len(set(keys))!=60 or set(keys)!={evaluator.job_key(j) for j in jobs}:
        raise ValueError('Development evidence does not include all60 fixed episode records')


def validate_plan(plan,check_exposure=True):
    if (plan['schema']!=1 or plan['selection_finalized'] is not True
            or plan['confirmation_previously_generated'] is not False
            or plan['smoke'] is not False or plan['split']!='confirmation'):
        raise ValueError('An explicit final selection and unused formal confirmation split are required')
    protocol_path=file_binding(plan['protocol']);protocol=read(protocol_path)
    if protocol_path!=R/'protocol.json':raise ValueError('Use the existing independent protocol')
    if file_binding(plan['scorer'])!=P/protocol['scorer'] or plan['scorer']['sha256']!=protocol['scorer_sha256']:
        raise ValueError('The original scorer binding changed')
    from physiologic_features import NORMALIZER_PATH
    if file_binding(plan['normalizer'])!=NORMALIZER_PATH.resolve():raise ValueError('Loop normalizer binding changed')
    runner=file_binding(plan['confirmation_runner'])
    runner.relative_to(R)
    if runner!=R/'evaluate_confirmation.py':
        raise ValueError('Use the separately reviewed evaluate_confirmation.py entrypoint')
    file_binding(plan['selection_evidence'])
    ids=[m['id'] for m in plan['methods']]
    comparators=plan['required_comparator_ids'];internal=plan['selected_internal_ids']
    if (not comparators or not internal or len(ids)!=len(set(ids))
            or any(not re.fullmatch('[a-z][a-z0-9_]{0,63}',x) for x in ids)
            or len(comparators)!=len(set(comparators)) or len(internal)!=len(set(internal))
            or set(comparators)&set(internal) or set(ids)!=set(comparators+internal)):
        raise ValueError('Freeze exactly the explicit required comparators and selected internal candidates')
    specs=case_specs(protocol)
    for method in plan['methods']:
        method_dependencies(method,runner);development_evidence(method)
    audit=prior_exposure_audit() if check_exposure else None
    return dict(plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),case_count=len(specs),
                case_spec_sha256=hashlib.sha256(canonical(specs)).hexdigest(),exposure_audit=audit)


@contextmanager
def lock(directory,timeout_seconds=30.):
    path=directory/'.gate.lock'
    deadline=time.monotonic()+timeout_seconds
    while True:
        try:
            descriptor=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            break
        except FileExistsError as error:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('Timed out waiting for confirmation gate lock; existing lock retained: '+str(path)) from error
            # Retry only lock acquisition, never a ticket or experiment. No stale-lock deletion.
            time.sleep(min(.05,remaining))
    try:
        os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
        yield
    finally:path.unlink()


def freeze(plan_path):
    plan=read(plan_path);result=validate_plan(plan)
    base=R/'confirmation';base.mkdir(exist_ok=True)
    with lock(base):
        if any(base.glob('*/freeze.json')):raise ValueError('A confirmation selection already exists; no replacement or second selection')
        # Recheck exposure after serializing competing freeze operations.
        result['exposure_audit']=prior_exposure_audit()
        directory=base/result['plan_sha256'];directory.mkdir(exist_ok=False)
        (directory/'runs').mkdir();(directory/'events').mkdir()
        write_new(directory/'freeze.json',dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),
                  plan=plan,**result))
        write_new(directory/'seal.json',dict(freeze_sha256=sha(directory/'freeze.json')))
    return directory/'freeze.json'


def load_freeze(path,revalidate=True):
    path=Path(path).resolve();path.relative_to((R/'confirmation').resolve())
    if path.name!='freeze.json':raise ValueError('Expected frozen selection manifest')
    frozen=read(path);digest=hashlib.sha256(canonical(frozen['plan'])).hexdigest()
    if digest!=frozen['plan_sha256'] or path.parent.name!=digest or read(path.parent/'seal.json')['freeze_sha256']!=sha(path):
        raise ValueError('Frozen selection or seal changed')
    if revalidate:validate_plan(frozen['plan'],check_exposure=False)
    return frozen


def events(directory):
    return [read(p) for p in sorted((directory/'events').glob('*.json'))]


def append_event(directory,value):
    folder=directory/'events'
    number=len(list(folder.glob('*.json')))+1
    temporary=folder/('.%04d.%s.tmp'%(number,uuid.uuid4().hex))
    write_new(temporary,dict(time_utc=datetime.now(timezone.utc).isoformat(),**value))
    # Callers hold the gate lock. Publish complete bytes without replacing any event.
    # On write/link failure retain the temporary evidence; readers only glob *.json.
    os.link(str(temporary),str(folder/('%04d.json'%number)))
    temporary.unlink()


def authorize(freeze_path,method_id):
    path=Path(freeze_path).resolve();frozen=load_freeze(path);directory=path.parent
    with lock(directory):
        prior=events(directory)
        if any(e['freeze_sha256']!=sha(path) for e in prior):raise ValueError('Freeze changed after prior authorization')
        if any(e.get('method_id')==method_id for e in prior):raise ValueError('No automatic retry, duplicate run or partial rerun')
        methods=[m for m in frozen['plan']['methods'] if m['id']==method_id]
        if len(methods)!=1:raise ValueError('Method not in the frozen comparison set')
        method=methods[0]
        import evaluate_candidates as evaluator
        jobs=[dict(spec,meals=evaluator.scenario(spec['seed'],spec['total_minutes'])) for spec in case_specs(read(R/'protocol.json'))]
        out=directory/'runs'/method_id;out.mkdir(exist_ok=False)
        ticket=dict(schema=1,freeze_path=str(path.relative_to(P)),freeze_sha256=sha(path),method_id=method_id,
            method=method,jobs=jobs,jobs_sha256=evaluator.json_sha(jobs),split='confirmation',smoke=False,
            planned_count=60,output_dir=str(out.relative_to(P)),automatic_retry_allowed=False,
            runner=frozen['plan']['confirmation_runner'],protocol=frozen['plan']['protocol'],
            scorer=frozen['plan']['scorer'],normalizer=frozen['plan']['normalizer'])
        write_new(out/'ticket.json',ticket)
        append_event(directory,dict(event='authorized',method_id=method_id,freeze_sha256=sha(path),
                     ticket_path=str((out/'ticket.json').relative_to(P)),ticket_sha256=sha(out/'ticket.json')))
    return out/'ticket.json'


def validate_ticket(ticket_path,revalidate=True):
    ticket_path=Path(ticket_path).resolve();ticket=read(ticket_path)
    freeze_path=project_path(ticket['freeze_path']);frozen=load_freeze(freeze_path,revalidate)
    out=project_path(ticket['output_dir'])
    if ticket_path!=out/'ticket.json' or out!=freeze_path.parent/'runs'/ticket['method_id']:
        raise ValueError('Ticket output directory changed')
    expected=next((m for m in frozen['plan']['methods'] if m['id']==ticket['method_id']),None)
    event=[e for e in events(freeze_path.parent) if e.get('method_id')==ticket['method_id'] and e['event']=='authorized']
    if (len(event)!=1 or event[0]['ticket_sha256']!=sha(ticket_path)
            or ticket['freeze_sha256']!=sha(freeze_path) or ticket['method']!=expected
            or ticket['planned_count']!=60 or ticket['smoke'] is not False or ticket['split']!='confirmation'):
        raise ValueError('Ticket differs from its single frozen authorization')
    return ticket


def start_run(ticket_path):
    """Consume one ticket exactly once, including failed/aborted starts."""
    ticket=validate_ticket(ticket_path);directory=project_path(ticket['freeze_path']).parent
    out=project_path(ticket['output_dir'])
    with lock(directory):
        if any(e.get('method_id')==ticket['method_id'] and e['event'] in ('started','finished') for e in events(directory)):
            raise ValueError('A started or finished confirmation ticket cannot run again')
        if {p.name for p in out.iterdir()}!={'ticket.json'}:
            raise ValueError('Confirmation output contains pre-existing data')
        write_new(out/'start.json',dict(ticket_sha256=sha(ticket_path),freeze_sha256=ticket['freeze_sha256'],
                                      method_id=ticket['method_id'],pid=os.getpid()))
        append_event(directory,dict(event='started',method_id=ticket['method_id'],freeze_sha256=ticket['freeze_sha256'],
            ticket_sha256=sha(ticket_path),start_sha256=sha(out/'start.json')))
    return ticket


def check_result(ticket,summary_path):
    import evaluate_candidates as evaluator
    import summarize_panels as aggregation
    out=project_path(ticket['output_dir']);summary_path=Path(summary_path).resolve();summary_path.relative_to(out)
    if summary_path!=out/'summary.json':raise ValueError('Expected the single fixed summary.json')
    manifest_path=out/'manifest.json';manifest=read(manifest_path)
    started=[e for e in events(project_path(ticket['freeze_path']).parent)
             if e.get('method_id')==ticket['method_id'] and e['event']=='started']
    if (len(started)!=1 or started[0]['ticket_sha256']!=sha(out/'ticket.json')
            or started[0]['start_sha256']!=sha(out/'start.json')):
        raise ValueError('Confirmation result lacks its unique execution start')
    summary=read(summary_path);episodes=summary['episodes'];jobs=ticket['jobs']
    expected={evaluator.job_key(j):j for j in jobs}
    if (summary['status'] not in ('completed','technical_failure') or summary['kind']!=ticket['method']['kind']
            or summary.get('method')!=ticket['method']['method']
            or summary['split']!='confirmation' or summary['smoke'] is not False
            or summary['count']!=60 or summary['planned_count']!=60 or len(episodes)!=60
            or len({e['key'] for e in episodes})!=60 or {e['key'] for e in episodes}!=set(expected)
            or summary['jobs_sha256']!=ticket['jobs_sha256']):
        raise ValueError('Confirmation must retain the entire single fixed60 case set')
    provenance=summary['provenance']
    if (provenance['freeze_sha256']!=ticket['freeze_sha256'] or provenance['ticket_sha256']!=sha(out/'ticket.json')
            or provenance['scorer_sha256']!=ticket['scorer']['sha256']
            or provenance['protocol_sha256']!=ticket['protocol']['sha256']
            or provenance['normalization_sha256']!=ticket['normalizer']['sha256']
            or provenance['worker_sha256']!=(ticket['method']['worker']['sha256'] if ticket['method']['worker'] else None)
            or provenance['config_sha256']!=method_config_sha(ticket['method'])
            or provenance['source_sha256']!=ticket['method']['source_sha256']
            or provenance['checkpoint_sha256']!=(ticket['method']['checkpoint']['sha256'] if ticket['method']['checkpoint'] else None)):
        raise ValueError('Confirmation output is not bound to the authorized model and scorer')
    if (provenance['manifest_sha256']!=sha(manifest_path) or manifest['jobs']!=jobs
            or manifest['jobs_sha256']!=ticket['jobs_sha256'] or manifest['split']!='confirmation'
            or manifest['kind']!=ticket['method']['kind'] or manifest.get('method')!=ticket['method']['method']
            or manifest['smoke'] is not False or manifest['confirmation'] is not True
            or manifest['tuning_split'] is not False or manifest['protocol']!=read(project_path(ticket['protocol']['path']))):
        raise ValueError('Confirmation manifest differs from its ticket or summary')
    bound_keys=('freeze_sha256','ticket_sha256','scorer_sha256','protocol_sha256','normalization_sha256',
                'worker_sha256','config_sha256','checkpoint_sha256','source_sha256')
    if any(manifest[key]!=provenance[key] for key in bound_keys):
        raise ValueError('Confirmation manifest/provenance binding mismatch')
    if any(manifest['artifact_sha256'].get(path)!=digest for path,digest in ticket['method']['artifact_sha256'].items()):
        raise ValueError('Confirmation manifest omits a frozen artifact')
    # Original scorer + timing + exact raw/summary checks, not a hand-written passed flag.
    aggregation.load_panel(out,ticket['method_id'])
    for episode in episodes:
        raw_path=(out/episode['raw_path']).resolve();raw_path.relative_to(out)
        if sha(raw_path)!=episode['raw_sha256']:raise ValueError('Confirmation raw evidence changed')
        raw=read(raw_path)
        if any(raw['evaluation_provenance'][key]!=provenance[key] for key in bound_keys+('manifest_sha256',)):
            raise ValueError('Raw confirmation provenance differs from the frozen run')
        if raw['job']!=expected[episode['key']]:raise ValueError('Raw trajectory job differs from fixed authorization')
        metrics=evaluator.summarize(raw['records'],792,raw['failure_reason'] is not None)
        if raw['metrics']!=metrics or episode['metrics']!=metrics:
            raise ValueError('Confirmation metrics do not match the unchanged original scorer')
        if not raw['raw_trajectory_available'] and (raw['records'] or not raw['unknown_tail']):
            raise ValueError('Absent trajectory must remain an empty unknown tail')
        if summary['status']=='completed' and (episode.get('technical_failure_reason') or not raw['raw_trajectory_available']):
            raise ValueError('Technical failure cannot be declared completed')
    artifact_hashes={}
    for path in out.rglob('*'):
        if path.is_file():
            path.resolve().relative_to(out)
            artifact_hashes[str(path.relative_to(P))]=sha(path)
    return summary['status'],artifact_hashes


def record_result(ticket_path,summary_path=None,failure_reason=None):
    if (summary_path is None)==(failure_reason is None):raise ValueError('Provide exactly a result summary or an explicit infrastructure failure')
    # Authenticate the immutable authorization first. Dependency drift must be
    # recorded as a failed run, never prevent us from preserving that failure.
    ticket=validate_ticket(ticket_path,revalidate=False);freeze_path=project_path(ticket['freeze_path']);directory=freeze_path.parent
    with lock(directory):
        if any(e.get('method_id')==ticket['method_id'] and e['event']=='finished' for e in events(directory)):
            raise ValueError('A finished run cannot be overwritten or retried')
        artifacts={};status='technical_failure';error=failure_reason
        try:
            validate_ticket(ticket_path)
            if summary_path is not None:status,artifacts=check_result(ticket,summary_path)
            elif not isinstance(failure_reason,str) or not failure_reason.strip():raise ValueError('Explicit failure reason required')
        except Exception as caught:
            error='Result validation failed: '+repr(caught)
        if error is not None:
            out=project_path(ticket['output_dir'])
            for path in out.rglob('*'):
                if path.is_file():
                    path.resolve().relative_to(out)
                    artifacts[str(path.relative_to(P))]=sha(path)
        result=dict(event='finished',method_id=ticket['method_id'],freeze_sha256=ticket['freeze_sha256'],
            ticket_sha256=sha(ticket_path),status=status,error=error,result_sha256=artifacts,
            complete_fixed_case_reporting=summary_path is not None and error is None,
            unknown_case_count=60 if summary_path is None else None,automatic_retry_allowed=False)
        append_event(directory,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='op',required=True)
    for name in ('validate','freeze'):
        sub=commands.add_parser(name);sub.add_argument('--plan',type=Path,required=True)
    sub=commands.add_parser('authorize');sub.add_argument('--freeze',type=Path,required=True);sub.add_argument('--method-id',required=True)
    sub=commands.add_parser('finish');sub.add_argument('--ticket',type=Path,required=True)
    group=sub.add_mutually_exclusive_group(required=True);group.add_argument('--summary',type=Path);group.add_argument('--failure-reason')
    args=parser.parse_args()
    if args.op=='validate':result=validate_plan(read(args.plan))
    elif args.op=='freeze':result=str(freeze(args.plan))
    elif args.op=='authorize':result=str(authorize(args.freeze,args.method_id))
    else:result=record_result(args.ticket,args.summary,args.failure_reason)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
