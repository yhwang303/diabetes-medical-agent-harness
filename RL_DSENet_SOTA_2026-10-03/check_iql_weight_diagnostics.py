"""Synthetic statistics/binding checks only; no Torch or real replay inference."""
import ast
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import diagnose_iql_weights as diagnostic

R = Path(__file__).resolve().parent
CHECKS = []
diagnostic.np = np


def check(name, condition):
    assert condition, name
    CHECKS.append(name)


def rejected(name, call):
    try:
        call()
    except (ValueError, FileNotFoundError, FileExistsError):
        CHECKS.append(name)
    else:
        raise AssertionError('Accepted invalid fixture: '+name)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('checks/iql_weight_diagnostic_mechanics_r1.json'))
    args=parser.parse_args()
    source = R/'diagnose_iql_weights.py'
    tree = ast.parse(source.read_text(), feature_version=(3,8))
    compile(tree, str(source), 'exec')
    check('python38_syntax', True)
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    agent_updates = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                     and n.func.attr=='update' and isinstance(n.func.value,ast.Name) and n.func.value.id=='agent']
    check('no_optimizer_update_backward_sampling_calls', not agent_updates and not set(calls)&{'step','backward','optimizers','sample','randperm','manual_seed'})
    one = diagnostic.statistics(np.zeros(100),np.zeros(100),np.ones(100))
    check('equal_weights_ESS100',one['effective_sample_size']==100 and one['ess_fraction']==1)
    check('equal_weights_top1percent_mass',one['top_one_percent_count']==1 and one['top_one_percent_weight_mass']==.01)
    check('exact_linear_quantiles',diagnostic.statistics(np.arange(5),np.arange(5)*3,np.ones(5))['advantage']['quantiles']['25']==1)
    w=np.array([0.,0.,0.,4.]);concentrated=diagnostic.statistics(np.zeros(4),np.zeros(4),w)
    check('concentrated_ESS1',concentrated['effective_sample_size']==1 and concentrated['top_one_percent_weight_mass']==1)
    check('zero_weights_undefined_mass_and_ESS',diagnostic.statistics(np.zeros(2),np.zeros(2),np.zeros(2))['effective_sample_size'] is None)
    check('empty_stratum_explicit',diagnostic.statistics([],[],[])['advantage']['quantiles'] is None)
    z=np.array([-1000,0,np.log(100),10],dtype=np.float32);w=np.exp(np.minimum(z,np.float32(np.log(100))))
    clipped=diagnostic.statistics(z/3,z,w)
    check('FP32_cap_and_underflow_count',clipped['weight_clip_count']==2 and clipped['weight_zero_count']==1)
    rejected('nonfinite_vector',lambda:diagnostic.statistics([np.nan],[0],[1]))
    rejected('wrong_vector_shape',lambda:diagnostic.statistics([0],[0,1],[1]))
    rejected('negative_weight',lambda:diagnostic.statistics([0],[0],[-1]))
    arrays=dict(action=np.array([-1,-.5,0,.5,1]),bg_mg_dl=np.array([53.,54.,70.,180.,181.]),episode_id=np.array([0,0,1,1,1]))
    eps=[dict(episode_id=0,source='natural'),dict(episode_id=1,source='ppo_first8')]
    strata,availability=diagnostic.groups(arrays,eps,5)
    check('action_bins_partition_boundaries',all(sum(mask[i] for mask in strata['action'].values())==1 for i in range(5)))
    check('BG_bins_partition_boundaries',[int(x.sum()) for x in strata['post_action_bg'].values()]==[1,1,2,1])
    check('stored_source_counts',[int(x.sum()) for x in strata['source'].values()]==[2,3])
    missing,available=diagnostic.groups({'action':arrays['action']},[{}],5)
    check('missing_labels_not_inferred',set(missing)=={'action'} and available['post_action_bg'].startswith('unavailable'))
    bad=dict(arrays,episode_id=np.array([0,0,2,2,2]))
    rejected('unbound_episode_id',lambda:diagnostic.groups(bad,eps,5))
    before={str(p):diagnostic.sha(p) for p in [R/n for n in ('iql_wide.py','prepare_iql_replay.py','train_iql_wide.py','iql_wide_worker.py','evaluate_candidates.py')]}
    config=json.loads((R/'configs/iql_wide.json').read_text())
    with tempfile.TemporaryDirectory(prefix='iql_weight_fixture_',dir=R/'checks') as tmp:
        root=Path(tmp);research=root/R.name;run=research/'results/IQL_wide';run.mkdir(parents=True)
        checkpoint=run/'policy_020000.pt';checkpoint.write_bytes(b'fixture, never torch loaded')
        template=research/'configs/iql_wide.json';write(template,config)
        source_dep=research/'fixture_source.py';source_dep.write_text('# fixture\n')
        replay=research/config['replay'];replay.mkdir(parents=True)
        hashes={str(p.relative_to(root)):diagnostic.sha(p) for p in [source_dep,template]}
        arrays_hash={}
        for name in ['frames','physiology','start','anchor','action','reward','terminal']:
            path=replay/(name+'.npy');path.write_bytes(b'fixture bytes, not numpy loaded');arrays_hash[path.name]=diagnostic.sha(path)
        write(replay/'episodes.json',[])
        manifest=dict(status='complete',config=config,config_sha256=diagnostic.sha(template),source_sha256=hashes,
            raw_data_sha256={'raw_fixture':'a'*64},selection={'train':'fixture'},episodes=280,transitions=2,
            natural_episodes=120,ppo_episodes=160,no_development_or_confirmation_data=True,
            identifiers_metadata_only=True,true_BG_reward_only=True,array_sha256=arrays_hash,
            episodes_sha256=diagnostic.sha(replay/'episodes.json'))
        provenance=dict(smoke=False,training_seed=260915,planned_updates=20000,episodes=280,transitions=2,
            config_sha256=diagnostic.sha(template),source_sha256=hashes,raw_data_sha256=manifest['raw_data_sha256'],
            data_selection=manifest['selection'],replay_path=str(replay.relative_to(root)))
        completion=dict(status='fixed_budget_completed',steps=20000,seed=260915,sample_visits=5120000,
            development_or_confirmation_used=False,final_checkpoint=checkpoint.name,
            checkpoints=[dict(path=checkpoint.name,step=20000,sha256=diagnostic.sha(checkpoint))])
        def reset():
            write(replay/'manifest.json',manifest)
            provenance['replay_manifest_sha256']=diagnostic.sha(replay/'manifest.json')
            write(run/'config.json',config);write(run/'provenance.json',provenance);write(run/'completion.json',completion)
        reset()
        with patch.multiple(diagnostic,R=research,P=root):
            call=lambda:diagnostic.binding(checkpoint,[source_dep])
            result=call();check('formal_bound_metadata_accepted',result[0]==config and len(result[4])>=14)
            rejected('relative_checkpoint_rejected',lambda:diagnostic.binding(Path('policy_020000.pt'),[source_dep]))
            rejected('intermediate_checkpoint_rejected',lambda:diagnostic.binding(run/'policy_010000.pt',[source_dep]))
            for label,where,field,value in [
                ('smoke_rejected','provenance.json','smoke',True),
                ('wrong_training_seed','provenance.json','training_seed',1),
                ('wrong_budget','completion.json','steps',19999),
                ('heldout_selection','completion.json','development_or_confirmation_used',True),
                ('running_training','completion.json','status','running'),
                ('changed_replay_binding','provenance.json','replay_manifest_sha256','0'*64),
                ('missing_required_source','provenance.json','source_sha256',{}),
                ('wrong_episode_mixture','provenance.json','episodes',279)]:
                path=run/where;bad=diagnostic.read(path);bad[field]=value;write(path,bad)
                rejected(label,call);reset()
            cp=checkpoint.read_bytes();checkpoint.write_bytes(cp+b'changed');rejected('changed_checkpoint_bytes',call);checkpoint.write_bytes(cp)
            path=replay/'action.npy';old=path.read_bytes();path.write_bytes(b'changed');rejected('changed_replay_array',call);path.write_bytes(old)
            old=source_dep.read_bytes();source_dep.write_bytes(old+b'changed');rejected('changed_source_bytes',call);source_dep.write_bytes(old)
            bad=deepcopy(manifest);bad['no_development_or_confirmation_data']=False;write(replay/'manifest.json',bad)
            provenance['replay_manifest_sha256']=diagnostic.sha(replay/'manifest.json');write(run/'provenance.json',provenance)
            rejected('nontraining_replay_even_if_rehashed',call);reset()
            state=dict(schema=1,kind='IQL_wide',step=20000,smoke=False,final=True,config=config,provenance=provenance)
            diagnostic.validate_checkpoint(state,config,provenance);check('embedded_payload_accepted',True)
            for field,value in [('step',5000),('smoke',True),('final',False),('config',{}),('provenance',{})]:
                bad=dict(state);bad[field]=value
                rejected('embedded_'+field+'_mismatch',lambda bad=bad:diagnostic.validate_checkpoint(bad,config,provenance))
            # Real CLI failure must retain an artifact and refuse overwriting it.
            target=research/'checks/failure.json';target.parent.mkdir()
            argv=['diagnose_iql_weights.py','--checkpoint',str(checkpoint),'--output',str(target)]
            with patch.object(sys,'argv',argv),patch.object(diagnostic,'run',side_effect=RuntimeError('synthetic unavailable runtime')):
                try:diagnostic.main()
                except RuntimeError:pass
                else:raise AssertionError('Failure hidden')
                check('failure_JSON_retained',diagnostic.read(target)['status']=='technical_failure')
                rejected('existing_output_refused',diagnostic.main)
    check('frozen_IQL_and_evaluator_unchanged',all(diagnostic.sha(Path(p))==digest for p,digest in before.items()))
    check('Torch_not_imported',not any(k=='torch' or k.startswith('torch.') for k in sys.modules))
    result=dict(status='passed',count=len(CHECKS),checks=CHECKS,fixture_only=True,
        checkpoint_deserialization=False,model_inference_executed=False,training_executed=False,
        remote_executed=False,diagnostic_source_sha256=diagnostic.sha(source),
        test_source_sha256=diagnostic.sha(Path(__file__).resolve()),frozen_source_sha256=before)
    output=diagnostic.inside(args.output if args.output.is_absolute() else R/args.output,R/'checks')
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='passed',count=len(CHECKS),diagnostic_source_sha256=result['diagnostic_source_sha256'])))


if __name__=='__main__':main()
