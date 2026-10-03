"""Audit all frozen confirmation runs using unchanged original aggregation.

Importing this module is standard-library only; report validation below never
loads a model, simulator, NumPy or scorer. The CLI's build step loads the original
aggregation module and really recomputes every raw score before writing output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

R=Path(__file__).resolve().parent
P=R.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value,ascii=True):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=ascii,allow_nan=False).encode()


def snapshot(path):
    data=Path(path).read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(),text=data.decode('utf-8'))


def validate_bundle(bundle):
    """Validate self-contained metadata snapshots, set completeness and code identity."""
    audit=bundle['confirmation_audit'];metadata=audit['metadata']
    require(audit['schema']==1 and audit['all_frozen_methods_reported'] is True,'Missing complete frozen-set audit')
    for name in ('confirmation_gate.py','evaluate_candidates.py','evaluate_confirmation.py',
                 'summarize_panels.py','summarize_confirmation.py','build_report.py'):
        require(audit['reporting_source_sha256'][name]==sha(R/name),'Confirmation audit source drift: '+name)
    require(bundle['provenance']['entrypoint_sha256']==sha(Path(__file__)), 'Confirmation aggregation entrypoint mismatch')
    def read(path):
        require(not Path(path).is_absolute() and '..' not in Path(path).parts,'Invalid metadata path')
        item=metadata[path]
        require(hashlib.sha256(item['text'].encode('utf-8')).hexdigest()==item['sha256'],'Metadata snapshot hash mismatch')
        return json.loads(item['text'])
    freeze_path=audit['freeze_path'];freeze=read(freeze_path);freeze_sha=metadata[freeze_path]['sha256']
    base=Path(freeze_path).parent
    seal=read(str(base/'seal.json'));plan=freeze['plan']
    digest=hashlib.sha256(canonical(plan,False)).hexdigest()
    require(seal['freeze_sha256']==freeze_sha and freeze['plan_sha256']==base.name==digest,
            'Frozen selection/seal identity mismatch')
    methods={m['id']:m for m in plan['methods']};mapping=audit['method_panels']
    require(len(methods)==len(plan['methods']) and set(methods)==set(mapping) and
            set(methods)==set(plan['required_comparator_ids']+plan['selected_internal_ids']) and
            set(mapping.values())==set(bundle['panels']) and len(set(mapping.values()))==len(mapping),
            'Confirmation omitted or added a frozen method')
    require(plan['selection_finalized'] is True and plan['confirmation_previously_generated'] is False and
            plan['split']=='confirmation' and plan['smoke'] is False,'Invalid frozen confirmation plan')
    for mid,method in methods.items():
        panel=bundle['panels'][mapping[mid]];paths=audit['methods'][mid]
        ticket=read(paths['ticket']);manifest=read(paths['manifest'])
        start=read(paths['start']);authorized=read(paths['authorized']);started=read(paths['started']);finished=read(paths['finished'])
        ticket_sha=metadata[paths['ticket']]['sha256'];manifest_sha=metadata[paths['manifest']]['sha256']
        expected_out=base/'runs'/mid
        require(paths['ticket']==str(expected_out/'ticket.json') and paths['start']==str(expected_out/'start.json') and
                paths['manifest']==str(expected_out/'manifest.json') and ticket['output_dir']==str(expected_out),
                'Confirmation output escaped its frozen run')
        require(ticket['freeze_path']==freeze_path and ticket['freeze_sha256']==freeze_sha and ticket['method']==method and
                ticket['method_id']==mid and ticket['split']=='confirmation' and ticket['smoke'] is False and
                ticket['planned_count']==len(ticket['jobs'])==60 and
                ticket['jobs_sha256']==hashlib.sha256(canonical(ticket['jobs'])).hexdigest(), 'Ticket binding mismatch')
        require(authorized['event']=='authorized' and started['event']=='started' and finished['event']=='finished' and
                all(e['method_id']==mid and e['freeze_sha256']==freeze_sha and e['ticket_sha256']==ticket_sha
                    for e in (authorized,started,finished)) and
                started['start_sha256']==metadata[paths['start']]['sha256'] and
                start['ticket_sha256']==ticket_sha and start['freeze_sha256']==freeze_sha,'Execution event mismatch')
        require(finished['complete_fixed_case_reporting'] is True and finished['error'] is None and
                finished['status']==panel['status'] and panel['status'] in ('completed','technical_failure'),
                'Unfinished or unauditable confirmation cannot be marked scored')
        require(panel['split']=='confirmation' and panel['smoke'] is False and panel['scoring_verified_exact'] is True and
                panel['episodes']==60 and panel['manifest_sha256']==manifest_sha and
                panel['controller']==method['kind'] and
                panel['checkpoint_sha256']==(method['checkpoint']['sha256'] if method['checkpoint'] else None) and
                panel['jobs_sha256']==ticket['jobs_sha256'] and
                panel['jobs']==sorted(ticket['jobs'],key=lambda j:'%s_p%02d_s%d'%(j['group'],j['patient'],j['seed'])),
                'Audited panel differs from its frozen method/jobs')
        require(manifest['jobs']==ticket['jobs'] and manifest['confirmation'] is True and manifest['tuning_split'] is False and
                manifest['freeze_sha256']==freeze_sha and manifest['ticket_sha256']==ticket_sha and
                manifest['source_sha256']==method['source_sha256'],'Manifest is not the frozen execution')
        outputs=finished['result_sha256']
        require(outputs[paths['manifest']]==manifest_sha and outputs[str(expected_out/'summary.json')]==panel['summary_sha256'] and
                all(outputs[c['raw_path']]==c['raw_sha256'] for c in panel['cases']), 'Finished event/raw/summary hashes differ')
        for name,digest in audit['reporting_source_sha256'].items():
            relative=str((R/name).relative_to(P))
            require(method['source_sha256'][relative]==digest,'Reporting source omitted from frozen method')
    return audit


def build(freeze_path,reference,output):
    import confirmation_gate as gate
    import summarize_panels as agg
    freeze_path=Path(freeze_path).resolve();frozen=gate.load_freeze(freeze_path)
    methods=frozen['plan']['methods'];ids=[m['id'] for m in methods]
    require(reference in ids,'Reference must be an explicit frozen method ID')
    output=Path(output)
    require(not output.is_absolute(),'Output must be a new checks/panels_confirmation_*.json')
    output=(R/output).resolve()
    require(output.parent==R/'checks' and output.name.startswith('panels_confirmation_') and output.suffix=='.json' and
            not output.exists(),'Refusing an invalid/existing confirmation report output')
    metadata={};events=[];panels={};audit_methods={};mapping={}
    def keep(path):
        path=Path(path).resolve();relative=str(path.relative_to(P));metadata[relative]=snapshot(path);return relative
    keep(freeze_path);keep(freeze_path.parent/'seal.json')
    for path in sorted((freeze_path.parent/'events').glob('*.json')):
        relative=keep(path);events.append((relative,json.loads(metadata[relative]['text'])))
    require(all(e.get('method_id') in ids for _,e in events),'Unexpected method in confirmation events')
    for method in methods:
        mid=method['id'];out=freeze_path.parent/'runs'/mid;ticket_path=out/'ticket.json'
        ticket=gate.validate_ticket(ticket_path)
        event_paths={}
        for event in ('authorized','started','finished'):
            matches=[(path,e) for path,e in events if e.get('method_id')==mid and e['event']==event]
            require(len(matches)==1,'Every frozen method needs one '+event+' event: '+mid)
            event_paths[event]=matches[0][0]
        finished=json.loads(metadata[event_paths['finished']]['text'])
        require(finished['error'] is None and finished['complete_fixed_case_reporting'] is True,
                'Frozen run lacks a complete auditable result: '+mid)
        status,output_hashes=gate.check_result(ticket,out/'summary.json')
        require(status==finished['status'] and output_hashes==finished['result_sha256'],
                'Result changed after its finished event: '+mid)
        name='confirmation_'+mid
        # These flags and numerical fields are returned by real original-score checks.
        panels[name]=agg.load_panel(out,name);mapping[mid]=name
        manifest=json.loads((out/'manifest.json').read_text())
        if method['kind'] in ('ppo_mean','world_ppo_mean'):
            panels[name]['deployment']={k:manifest[k] for k in ('deployment_rule','underlying_policy_kind',
                'same_checkpoint_new_deployment_ablation','original_preregistration','new_external_method')}
        audit_methods[mid]=dict(ticket=keep(ticket_path),start=keep(out/'start.json'),manifest=keep(out/'manifest.json'),**event_paths)
    ref=mapping[reference]
    payload=dict(schema=1,reference=ref,panel_order=list(panels),panels=panels,
        comparisons={name:agg.compare(panel,panels[ref]) for name,panel in panels.items() if name!=ref},
        methodology=dict(bg_primary=True,cgm_secondary=True,scoring_dt_minutes=5,scorer=str(agg.SCORER.relative_to(P)),
            scorer_sha256=sha(agg.SCORER),trajectory_glucose_sd_ddof=0,between_patient_sd_ddof=1,
            raw_and_summary_score_verification='Every field exactly equal to unchanged control_metrics.summarize; no numeric tolerance',
            aggregation='Unchanged summarize_panels.patient_tables: within-patient case means, then patient sample SD',
            incomplete_rule='Failed/incomplete runs remain Observed; never complete-method ranking',
            decision_tolerance=agg.TOL,bootstrap=dict(seed=agg.BOOTSTRAP_SEED,replicates=agg.BOOTSTRAP_REPLICATES,
                seed_is_training_seed=False,unit='paired patient cluster',descriptive_no_multiplicity_correction=True)),
        provenance=dict(script_sha256=sha(Path(agg.__file__)),script_role='unchanged aggregation module',
            entrypoint_sha256=sha(Path(__file__)),protocol_sha256=sha(R/'protocol.json'),
            numpy_version=agg.np.__version__,python_version=sys.version),
        confirmation_audit=dict(schema=1,all_frozen_methods_reported=True,freeze_path=str(freeze_path.relative_to(P)),
            method_panels=mapping,methods=audit_methods,metadata=metadata,
            reporting_source_sha256={name:sha(R/name) for name in ('confirmation_gate.py','evaluate_candidates.py',
                'evaluate_confirmation.py','summarize_panels.py','summarize_confirmation.py','build_report.py')}))
    validate_bundle(payload)
    with output.open('x') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--reference',required=True,help='Method ID from the frozen plan')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(build(args.freeze,args.reference,args.output))


if __name__=='__main__':main()
