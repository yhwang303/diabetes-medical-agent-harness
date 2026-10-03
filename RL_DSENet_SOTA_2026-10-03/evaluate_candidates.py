"""Common development runner; original simulator/scorer, no confirmation access.

`legacy` means the frozen P03/D05/D06 bounded actor, not the unrelated old
NumPy external baseline. No model receives job metadata, nominal basal, BG,
future meals, CR or CF. PPO/world/world-PPO/IQL receive only op=evaluate/history/anchors and never
act/update, so PPO's training buffer is not populated. World control is MPC,
not RL. Smoke uses the same two
predeclared development jobs for every method, retaining the full duration.
Retained external methods keep their original absolute-rate workers and are
explicitly labelled frozen+projection onto the new common action support.
Their opaque stream handles/scenario seeds manage RNGs, not network inputs.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import hashlib
import json
import multiprocessing as mp
import selectors
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
B = P / 'RL_DSENet_2026-09-17'
E = P / 'RL_DSENet_公平低糖_2026-09-21'
SCORER = P / 'RL_DITR创新_2026-09-16/control_metrics.py'
sys.path.insert(0, str(E))
from ppo_env import environment_worker, scenario
from control_metrics import summarize
from controller_baselines import make_controller
from physiologic_features import NORMALIZER_PATH, normalization

COMMON_SOURCES = [Path(__file__), R/'controller_baselines.py', R/'physiologic_features.py',
                  E/'ppo_env.py', E/'evaluate.py', E/'brake.py',
                  P/'RL进阶对比_2026-09-15/observable_history.py', SCORER]
RETAINED_METHODS = ('bc','td3bc','iql','rebrac','fql','lom','gfp','ditr')
MEAN_KINDS = {'ppo_mean':'ppo','world_ppo_mean':'world_ppo'}
MEAN_RULE = 'probability_mean_of_capped_executable_grid'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def make_jobs(protocol, split, smoke=False):
    if split not in ('development', 'world_validation'):
        raise ValueError('Confirmation requires a separately frozen final runner')
    if protocol['warmup_minutes'] != 360 or protocol['total_minutes'] != 4320:
        raise ValueError('The original environment requires the frozen 360/4320 minute contract')
    seeds = protocol[split + '_scenario_seeds']
    if set(seeds) & set(protocol['confirmation_scenario_seeds']):
        raise ValueError('Evaluation seeds overlap the reserved confirmation split')
    jobs = [dict(patient=p, seed=s, group='bolus_%.1f' % f, bolus_factor=f,
                 total_minutes=protocol['total_minutes'],
                 meals=scenario(s, protocol['total_minutes']))
            for p in protocol['patients'] for f in protocol['bolus_factors'] for s in seeds]
    if len(jobs) != 60:
        raise ValueError('Expected the frozen 10 patients x 3 factors x 2 scenarios')
    if smoke:
        jobs = [jobs[0], next(j for j in jobs if j['patient'] == 9 and j['bolus_factor'] == 1.2)]
    return jobs


def job_key(job):
    return '%s_p%02d_s%d' % (job['group'], job['patient'], job['seed'])


def warmup_anchor(history):
    h = np.asarray(history, dtype=np.float32)
    if h.shape != (72, 22) or not np.isfinite(h).all() or h[-1, 6] <= .5:
        raise ValueError('A finite observed warmup basal history is required')
    mean, scale, _ = normalization()
    anchor = float((float(h[-1, 1]) * scale[1] + mean[1]) * 12)
    if not 0 < anchor <= 20:
        raise ValueError('Invalid observed warmup anchor')
    return anchor


def policy_payload(kind, histories, anchors, method=None, episode_handles=None, scenario_seeds=None):
    kind=MEAN_KINDS.get(kind,kind)
    if kind in ('ppo', 'world', 'world_ppo', 'iql_wide'):
        return dict(op='evaluate', history=histories.tolist(), anchors=anchors.tolist())
    if kind == 'legacy':
        return dict(history=histories.tolist(), anchor_u_h=anchors.tolist())
    if kind == 'retained':
        if method not in RETAINED_METHODS:
            raise ValueError('Unknown retained method')
        if method == 'ditr':
            return dict(history=histories.tolist())
        if (episode_handles is None or scenario_seeds is None
                or len(episode_handles)!=len(histories) or len(scenario_seeds)!=len(histories)
                or len(set(episode_handles))!=len(histories)
                or any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in episode_handles)):
            raise ValueError('One opaque RNG handle and scenario seed are required per episode')
        return dict(history=histories.tolist(),case_keys=episode_handles,scenario_seeds=scenario_seeds)
    raise ValueError('Only Torch methods use the JSON worker')


def validate_actions(kind, actions, anchors):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != anchors.shape or not np.isfinite(actions).all():
        raise ValueError('Wrong shape or nonfinite policy actions')
    lower = np.maximum(0, anchors - .25) if kind == 'legacy' else np.zeros_like(anchors)
    upper = np.minimum(20, anchors + .25) if kind == 'legacy' else np.minimum(20, 2*anchors)
    if np.any(actions < lower - 1e-6) or np.any(actions > upper + 1e-6):
        raise ValueError('Policy exceeded its declared frozen action support')
    # Validation only: do not silently project or replace a failing action.
    if np.any(actions < 0) or np.any(actions > 20):
        raise ValueError('Policy exceeded simulator limits')
    if kind == 'world_ppo':
        grid = np.minimum(20, anchors.astype(np.float32)[:,None] * (np.arange(9,dtype=np.float32)/4)[None])
        if np.any(np.abs(actions[:,None]-grid).min(-1)>1e-6):
            raise ValueError('World-PPO action is outside its fixed nine-point grid')
    return actions


def retained_definition(method):
    if method not in RETAINED_METHODS:
        raise ValueError('Retained evaluation requires an explicit --method')
    path=R/'configs/retained_methods.json'; config=json.loads(path.read_text())
    original=P/config['final_manifest']
    if sha(original)!=config['final_manifest_sha256'] or config['training_seed']!=260915:
        raise ValueError('Original retained final manifest/seed mismatch')
    if config['normalization_sha256']!=sha(NORMALIZER_PATH) or config['deployment_label']!='frozen+projection':
        raise ValueError('Retained normalization or deployment-label mismatch')
    if (set(config['methods'])!=set(RETAINED_METHODS) or config['action']!=dict(
            original_worker_unit='U/h',original_worker_bounds=[0,20],
            projection='clip(raw_u_h,0,min(20,2*observed_warmup_anchor_u_h))',network_action_remapping=False)):
        raise ValueError('Retained common-task projection configuration changed')
    entry=config['methods'][method]
    old=next(m for m in json.loads(original.read_text())['methods'] if m['key']==method)
    if any(entry[k]!=old[k] for k in ('key','checkpoint','checkpoint_sha256','mode','shared_worker')):
        raise ValueError('Retained checkpoint does not match the original final manifest')
    worker=P/entry['worker']
    expected_worker=P/'RL_DITR创新_2026-09-16/policy_worker.py' if method=='ditr' else E/'shared_worker.py'
    if worker!=expected_worker or entry['worker_args']!=(['--mode','beam'] if method=='ditr' else []):
        raise ValueError('Retained method must use its original final GPU worker')
    if entry['worker'] not in entry['source_sha256'] or entry['label']!=old['label']+' frozen+projection':
        raise ValueError('Retained worker source binding/deployment label missing')
    for relative,digest in {entry['checkpoint']:entry['checkpoint_sha256'],
                            **entry['source_sha256'],**entry['metadata_sha256']}.items():
        source=(P/relative).resolve();source.relative_to(P)
        if sha(source)!=digest:
            raise ValueError('Retained dependency hash mismatch: '+relative)
    return config,entry


def project_retained_actions(raw, anchors):
    """Explicit task adaptation after the unchanged absolute-rate worker output."""
    raw=np.asarray(raw,dtype=np.float64);anchors=np.asarray(anchors,dtype=np.float64)
    if raw.shape!=anchors.shape or not np.isfinite(raw).all() or np.any((raw<0)|(raw>20)):
        raise ValueError('Retained worker must return finite original 0..20 U/h actions')
    if not np.isfinite(anchors).all() or np.any((anchors<=0)|(anchors>20)):
        raise ValueError('Invalid observed anchor for common projection')
    requested=np.clip(raw,0,np.minimum(20,2*anchors))
    return requested,requested!=raw,requested-raw


def projection_summary(decisions):
    count=len(decisions);projected=sum(d['projected'] for d in decisions)
    differences=[abs(d['projection_delta_u_h']) for d in decisions]
    return dict(scope='requested control decisions; independent of pump quantization',
                decisions=count,projected_decisions=projected,
                projection_rate_pct=100*projected/count if count else None,
                mean_absolute_projection_u_h=float(np.mean(differences)) if count else None,
                max_absolute_projection_u_h=max(differences) if count else None)


def episode_handle(index):
    return hashlib.sha256(('retained-episode-stream-'+str(index)).encode()).hexdigest()


def iql_definition(checkpoint, config, allow_smoke=False):
    """Read metadata without Torch; the worker also checks checkpoint contents."""
    from prepare_iql_replay import load_config, source_paths
    if not checkpoint.is_absolute() or not config.is_absolute():
        raise ValueError('IQL checkpoint and run config must be absolute')
    run=config.parent.resolve(); run.relative_to((R/'results').resolve())
    if config.name!='config.json' or checkpoint.parent.resolve()!=run:
        raise ValueError('IQL requires its checkpoint and config.json in the same run')
    c=load_config(config)
    template=R/'configs/iql_wide.json'
    if c!=json.loads(template.read_text()):
        raise ValueError('IQL run configuration differs from the fixed training configuration')
    provenance_path=run/'provenance.json'; completion_path=run/'completion.json'
    bound=json.loads(provenance_path.read_text()); completed=json.loads(completion_path.read_text())
    smoke=bound['smoke']
    if type(smoke) is not bool or (smoke and not allow_smoke):
        raise ValueError('IQL smoke weights require an explicitly smoke evaluation')
    steps=4 if smoke else 20000
    filename='smoke_step4.pt' if smoke else 'policy_020000.pt'
    if (checkpoint.name!=filename or bound['training_seed']!=260915
            or bound['planned_updates']!=steps or bound['config_sha256']!=sha(template)
            or completed['status']!=('smoke_completed' if smoke else 'fixed_budget_completed')
            or completed['steps']!=steps or completed['seed']!=260915
            or completed['sample_visits']!=steps*c['batch_size']
            or completed['final_checkpoint']!=(None if smoke else filename)
            or completed['development_or_confirmation_used'] is not False):
        raise ValueError('IQL checkpoint, completed budget or training selection mismatch')
    selected=[item for item in completed['checkpoints'] if item['path']==filename]
    if len(selected)!=1 or selected[0]['step']!=steps or selected[0]['sha256']!=sha(checkpoint):
        raise ValueError('IQL completed checkpoint hash mismatch')
    sources=source_paths(c)+[template]
    required={str(path.resolve().relative_to(P)) for path in sources}
    if not required.issubset(bound['source_sha256']):
        raise ValueError('IQL provenance omits required source/config bindings')
    for relative,digest in bound['source_sha256'].items():
        path=(P/relative).resolve(); path.relative_to(P)
        if sha(path)!=digest:
            raise ValueError('IQL training source hash mismatch: '+relative)
        sources.append(path)
    replay=(P/bound['replay_path']).resolve(); replay.relative_to((R/'data').resolve())
    if replay!=(R/c['replay']).resolve():
        raise ValueError('IQL replay path differs from its fixed config')
    replay_manifest=replay/'manifest.json'
    if sha(replay_manifest)!=bound['replay_manifest_sha256']:
        raise ValueError('IQL replay manifest hash mismatch')
    replay_bound=json.loads(replay_manifest.read_text())
    if (replay_bound['status']!='complete' or replay_bound['config']!=c
            or replay_bound['source_sha256']!=bound['source_sha256']
            or replay_bound['raw_data_sha256']!=bound['raw_data_sha256']
            or replay_bound['selection']!=bound['data_selection']
            or replay_bound['episodes']!=bound['episodes']):
        raise ValueError('IQL replay and training provenance disagree')
    protocol=json.loads((R/'protocol.json').read_text())
    if P/protocol['scorer']!=SCORER or sha(SCORER)!=protocol['scorer_sha256']:
        raise ValueError('IQL independent scorer binding mismatch')
    artifacts=[checkpoint,config,template,provenance_path,completion_path,replay_manifest]
    return c,bound,sources,artifacts


def validate_iql_ready(ready, manifest):
    if (ready.get('kind')!='IQL_wide' or ready.get('state_dim')!=1613
            or ready.get('deterministic') is not True
            or ready.get('checkpoint_sha256')!=manifest['checkpoint_sha256']
            or ready.get('config_sha256')!=manifest['config_sha256']
            or ready.get('provenance_sha256')!=manifest['policy_training_provenance_sha256']
            or ready.get('smoke')!=manifest['iql_training_smoke']):
        raise ValueError('IQL worker loaded a different checkpoint/config/provenance or interface')


def validate_iql_response(response, actions, anchors):
    normalized=np.asarray(response['normalized_actions'],dtype=np.float64)
    capped=np.asarray(response['global_cap_applied'])
    duplicate=np.asarray(response['actions_u_h'],dtype=np.float64)
    if (normalized.shape!=anchors.shape or not np.isfinite(normalized).all()
            or np.any(np.abs(normalized)>1) or capped.shape!=anchors.shape or capped.dtype.kind!='b'
            or duplicate.shape!=anchors.shape or not np.array_equal(actions,duplicate)):
        raise ValueError('Invalid IQL continuous-action diagnostics')
    raw=anchors*(normalized+1)
    if (not np.allclose(actions,np.minimum(raw,20),rtol=0,atol=1e-6)
            or not np.array_equal(capped,raw>20)):
        raise ValueError('IQL normalized action and intrinsic global-cap mapping disagree')
    return normalized,capped


def policy_evaluation_binding(kind, checkpoint, config, allow_smoke=False):
    """Bind a completed policy update without waiting for the remaining run."""
    kind=MEAN_KINDS.get(kind,kind)
    run=config.parent.resolve(); run.relative_to((R/'results').resolve())
    c=json.loads(config.read_text()); sources=[]; templates=[]
    if (config.name!='config.json' or checkpoint.parent.resolve()!=run
            or c['name']!=run.name or not checkpoint.stem.startswith('policy_iter')):
        raise ValueError('Policy evaluation needs its same-run checkpoint and config.json')
    number=checkpoint.stem[len('policy_iter'):]
    if not number.isdigit() or checkpoint.name!='policy_iter%02d.pt'%int(number):
        raise ValueError('Unexpected policy checkpoint filename')
    iteration=int(number)
    provenance_path=run/'provenance.json'; training=json.loads(provenance_path.read_text())
    if training['config']!=c or training['training_seed']!=260915:
        raise ValueError('Policy run config and training provenance disagree')
    if kind=='ppo':
        # This older trainer has no run_mode field; match its complete frozen
        # config (apart from the output name), not a name containing "smoke".
        modes=[]
        for mode,filename in (('formal','ppo_wide_risk.json'),('smoke','ppo_wide_smoke.json')):
            template=R/'configs'/filename; expected=json.loads(template.read_text())
            templates.append(template)
            if {k:v for k,v in c.items() if k!='name'}=={k:v for k,v in expected.items() if k!='name'}:
                modes.append(mode)
        if len(modes)!=1 or training['reward_BG_not_inference_input'] is not True:
            raise ValueError('PPO configuration is not the frozen formal or smoke contract')
        mode=modes[0]
        sources=[R/'train_wide.py',R/'ppo_wide_worker.py',R/'physiologic_features.py',E/'ppo_env.py']
        if set(training['source_sha256'])!={p.name for p in sources}:
            raise ValueError('PPO training provenance omits or changes its declared sources')
        if any(training['source_sha256'][p.name]!=sha(p) for p in sources):
            raise ValueError('PPO training source hash mismatch')
    elif kind=='world_ppo':
        from train_world_policy import validate_config
        validate_config(c,json.loads((R/'protocol.json').read_text()),bound=True)
        mode=c['run_mode']
        if Path(c['output_dir']).resolve()!=run:
            raise ValueError('World-PPO resolved output directory mismatch')
    else:
        raise ValueError('Only PPO policies have iteration-history bindings')
    if not 1<=iteration<=c['iterations']:
        raise ValueError('Policy iteration exceeds the declared training budget')
    if not allow_smoke and (mode!='formal' or c['development_checkpoints']!=[8,16,32,40]
                            or iteration not in (8,16,32,40)):
        raise ValueError('Formal policy evaluation requires a preregistered formal checkpoint')
    # A running trainer may append later rows. Freeze only the complete prefix
    # through this update; later writes neither block nor change this binding.
    history=run/'history.jsonl'; prefix=[]
    with history.open('rb') as stream:
        for expected_iteration in range(1,iteration+1):
            line=stream.readline()
            if not line.endswith(b'\n'):
                raise ValueError('Checkpoint has no complete training-history record')
            record=json.loads(line)
            if type(record['iteration']) is not int or record['iteration']!=expected_iteration:
                raise ValueError('Training-history iterations are not a contiguous completed prefix')
            prefix.append(line)
    if (not Path(record['checkpoint']).is_absolute()
            or Path(record['checkpoint']).resolve()!=checkpoint.resolve()
            or record['checkpoint_sha256']!=sha(checkpoint)):
        raise ValueError('Policy checkpoint is not bound to its completed training-history update')
    prefix=b''.join(prefix)
    binding=dict(training_mode=mode,iteration=iteration,checkpoint_sha256=record['checkpoint_sha256'],
        config_sha256=sha(config),training_provenance_sha256=sha(provenance_path),
        training_history=str(history),history_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        history_prefix_bytes=len(prefix),checkpoint_record=record)
    return binding,sources,[config,checkpoint,provenance_path]+templates,prefix


def world_evaluation_binding(provenance_path, training, allow_smoke=False):
    completed=json.loads((provenance_path.parent/'completion.json').read_text())
    c=training['config']
    if (type(c['budget_override']) is not bool or type(completed['budget_override']) is not bool
            or any(completed[k]!=c[k] for k in ('steps','configured_steps','budget_override'))):
        raise ValueError('World completed budget and training config disagree')
    if not allow_smoke and (c['steps']!=4000 or c['configured_steps']!=4000
                            or c['budget_override'] is not False):
        raise ValueError('Formal evaluation requires the completed 4000-step world, without a budget override')
    return dict(training_provenance_sha256=sha(provenance_path),
                completion_sha256=sha(provenance_path.parent/'completion.json'),
                completed_steps=completed['steps'],configured_steps=completed['configured_steps'],
                budget_override=completed['budget_override'])


def validate_policy_ready(ready, binding, checkpoint, config):
    if (type(ready.get('iteration')) is not int or ready['iteration']!=binding['iteration']
            or sha(checkpoint)!=binding['checkpoint_sha256'] or sha(config)!=binding['config_sha256']
            or sha(config.parent/'provenance.json')!=binding['training_provenance_sha256']):
        raise ValueError('Worker policy iteration or bytes differ from the bound training-history update')


def validate_mean_ready(ready, kind, manifest):
    family='wide' if kind=='ppo_mean' else 'world'
    binding=manifest['policy_evaluation_binding']
    expected_sources=[R/'ppo_mean_worker.py',R/('ppo_wide_worker.py' if family=='wide' else 'ppo_world_worker.py')]
    expected_sources={str(p.relative_to(P)):manifest['source_sha256'][str(p.relative_to(P))] for p in expected_sources}
    if (ready.get('kind')!='ppo_mean' or ready.get('family')!=family
            or ready.get('feature_dim')!=(334 if family=='wide' else 195)
            or ready.get('deployment_rule')!=MEAN_RULE or ready.get('original_deployment_rule')!='argmax'
            or ready.get('source_sha256')!=expected_sources
            or any(ready.get(k)!=binding[k] for k in ('checkpoint_sha256','config_sha256',
                    'training_provenance_sha256','history_prefix_sha256','history_prefix_bytes'))
            or any(ready.get(k) is not False for k in ('training_enabled','rollout_buffer_retained',
                    'random_action_sampling','original_preregistration'))
            or any(ready.get(k) is not True for k in ('same_checkpoint_new_deployment_ablation',
                    'proposed_after_ppo08_development_results'))):
        raise ValueError('Mean-deployment worker binding, family or deployment rule mismatch')


def validate_mean_response(response, actions, anchors):
    """Independently reconstruct the executable-grid mean, never rescale it."""
    probabilities=np.asarray(response['probabilities'],dtype=np.float64)
    grid=np.asarray(response['actual_action_grid_u_h'],dtype=np.float64)
    # Both unchanged family feature encoders expose their float32 anchor tensor.
    observed=np.asarray(anchors,dtype=np.float32).astype(np.float64)
    ratios=np.arange(9,dtype=np.float64)/4
    expected_grid=np.minimum(20,observed[:,None]*ratios[None])
    if (probabilities.shape!=(len(anchors),9) or not np.isfinite(probabilities).all()
            or np.any((probabilities<0)|(probabilities>1))
            or not np.allclose(probabilities.sum(-1),1,rtol=0,atol=1e-10)
            or grid.shape!=expected_grid.shape or not np.isfinite(grid).all()
            or not np.allclose(grid,expected_grid,rtol=0,atol=1e-10)):
        raise ValueError('Mean deployment has invalid probabilities or executable action grid')
    expected=(probabilities*expected_grid).sum(-1)
    duplicate=np.asarray(response['action_mean'],dtype=np.float64)
    chosen=np.asarray(response['argmax_indices'])
    if (not np.allclose(actions,expected,rtol=0,atol=1e-10)
            or duplicate.shape!=actions.shape or not np.array_equal(duplicate,actions)
            or chosen.shape!=anchors.shape or chosen.dtype.kind not in 'iu' or np.any((chosen<0)|(chosen>=9))):
        raise ValueError('Executed mean or original argmax diagnostics disagree with the distribution')
    row=np.arange(len(anchors)); selected=probabilities[row,chosen]
    if np.any(selected<probabilities.max(-1)-1e-12):
        raise ValueError('Reported argmax does not select a maximum-probability action')
    logp=np.zeros_like(probabilities); np.log(probabilities,out=logp,where=probabilities>0)
    ordered=np.sort(probabilities,axis=-1)
    ratio_mean=probabilities@ratios
    expected_fields=dict(argmax_actions_u_h=expected_grid[row,chosen],argmax_probabilities=selected,
        expected_ratio=ratio_mean,raw_ratio_mean_rate=observed*ratio_mean,
        hold_probability=probabilities[:,4],probability_margin_top2=ordered[:,-1]-ordered[:,-2],
        entropy=-(probabilities*logp).sum(-1),mean_minus_argmax_u_h=actions-expected_grid[row,chosen],
        arithmetic_boundary_adjustment_u_h=actions-expected)
    for key,value in expected_fields.items():
        reported=np.asarray(response[key],dtype=np.float64)
        if reported.shape!=anchors.shape or not np.isfinite(reported).all() or not np.allclose(reported,value,rtol=0,atol=1e-10):
            raise ValueError('Mean-deployment audit field mismatch: '+key)
    if np.any(np.abs(response['arithmetic_boundary_adjustment_u_h'])>1e-12):
        raise ValueError('Mean deployment introduced more than floating-point boundary adjustment')
    fields=['probabilities','actual_action_grid_u_h','argmax_indices']+list(expected_fields)
    return [{key:response[key][i] for key in fields} for i in range(len(anchors))]


def dependencies(kind, checkpoint, config, method=None, allow_smoke=False):
    if kind in MEAN_KINDS:
        family=MEAN_KINDS[kind]
        sources,artifacts,_,checkpoint=dependencies(family,checkpoint,config,method,allow_smoke)
        worker=R/'ppo_mean_worker.py'
        command=[str(P/'.venv-native/bin/python'),'-u',str(worker),
                 '--family','wide' if family=='ppo' else 'world',
                 '--config',str(config),'--checkpoint',str(checkpoint)]
        return sorted(set(sources+[worker])),artifacts,command,checkpoint
    sources = list(COMMON_SOURCES)
    artifacts = [R/'protocol.json', NORMALIZER_PATH]
    command = None
    if kind == 'iql_wide':
        _,bound,iql_sources,iql_artifacts=iql_definition(checkpoint,config,allow_smoke)
        sources+=iql_sources; artifacts+=iql_artifacts
        command=[str(P/'.venv-native/bin/python'),'-u',str(R/'iql_wide_worker.py'),
                 '--checkpoint',str(checkpoint),'--device','cuda']
        if bound['smoke']:
            command+=['--allow-smoke']
    elif kind == 'retained':
        retained,entry=retained_definition(method)
        checkpoint=P/entry['checkpoint']
        sources += [P/relative for relative in entry['source_sha256']]
        artifacts += [R/'configs/retained_methods.json',P/retained['final_manifest'],checkpoint]
        artifacts += [P/relative for relative in entry['metadata_sha256']]
        command=[str(P/'.venv-native/bin/python'),'-u',str(P/entry['worker']),
                 '--checkpoint',str(checkpoint)]+entry['worker_args']
    elif kind in ('legacy', 'ppo'):
        selected_path = R/'checks/frozen_selected_version.json'
        selected = json.loads(selected_path.read_text())
        world_config_path = B/'results/D05_selected_world/config.json'
        world_config = json.loads(world_config_path.read_text())
        frozen_dependencies = {key: (B/item['path'], item['sha256'])
                               for key, item in selected['weights'].items() if key != 'policy'}
        frozen_dependencies['context'] = (P/world_config['context_checkpoint'], world_config['context_sha256'])
        if kind == 'legacy':
            fixed = selected['weights']['policy']
            checkpoint = (B/fixed['path']).resolve()
            frozen_dependencies['policy'] = (checkpoint, fixed['sha256'])
        for label, (path, expected) in frozen_dependencies.items():
            if sha(path) != expected:
                raise ValueError('Frozen dependency hash mismatch: ' + label)
            artifacts.append(path)
        artifacts += [selected_path, B/'results/D06_selected_policy/config.json', world_config_path]
        worker = R/'ppo_wide_worker.py' if kind == 'ppo' else B/'bounded_policy_worker.py'
        sources += [worker, B/'world_model.py', B/'policy_bounded.py', B/'forecast_model.py',
                    P/'RL_DITR创新_2026-09-16/response_operator.py', P/'RL_DITR创新_2026-09-16/ditr_model.py']
        sources += sorted((B/'dsenet').rglob('*.py'))
        command = [str(P/'.venv-native/bin/python'), '-u', str(worker), '--checkpoint', str(checkpoint)]
        if kind == 'ppo':
            _,policy_sources,policy_artifacts,_=policy_evaluation_binding(kind,checkpoint,config,allow_smoke)
            sources+=policy_sources; artifacts+=policy_artifacts
            command += ['--config', str(config)]
            artifacts += [checkpoint, config]
        else:
            command += ['--mode', 'actor']
    elif kind == 'world':
        from world_control_worker import inspect_provenance, load_control_config
        load_control_config(config)
        world_path,world_training,world_sources,world_artifacts=inspect_provenance(checkpoint)
        world_evaluation_binding(world_path,world_training,allow_smoke)
        sources += world_sources + [R/'world_control_worker.py']
        artifacts += world_artifacts + [config]
        command = [str(P/'.venv-native/bin/python'), '-u', str(R/'world_control_worker.py'),
                   '--checkpoint', str(checkpoint), '--config', str(config)]
    elif kind == 'world_ppo':
        from train_world_policy import validate_config
        from world_control_worker import inspect_provenance
        protocol = json.loads((R/'protocol.json').read_text())
        c = json.loads(config.read_text())
        validate_config(c, protocol, bound=True)
        _,_,policy_artifacts,_=policy_evaluation_binding(kind,checkpoint,config,allow_smoke)
        artifacts+=policy_artifacts
        run = config.parent.resolve()
        run.relative_to((R/'results').resolve())
        number = checkpoint.stem[len('policy_iter'):]
        if (config.name!='config.json' or not run.name.startswith('PPO_world')
                or Path(c['output_dir']).resolve()!=run or c['name']!=run.name
                or checkpoint.parent.resolve()!=run or checkpoint.suffix!='.pt'
                or not checkpoint.stem.startswith('policy_iter') or not number.isdigit()
                or not 1<=int(number)<=c['iterations']):
            raise ValueError('World-PPO needs its resolved run config and same-run policy_iter checkpoint')
        if P/protocol['scorer']!=SCORER or sha(SCORER)!=protocol['scorer_sha256']:
            raise ValueError('World-PPO independent scorer binding mismatch')
        wp = Path(c['world_checkpoint'])
        provenance_path, world_provenance, world_sources, world_artifacts = inspect_provenance(wp)
        world_evaluation_binding(provenance_path,world_provenance,allow_smoke)
        for key,path in (('world_checkpoint_sha256',wp),
                         ('world_training_provenance_sha256',provenance_path),
                         ('world_completion_sha256',provenance_path.parent/'completion.json')):
            if c[key]!=sha(path):
                raise ValueError('Resolved World-PPO dependency hash mismatch: '+key)
        if (world_provenance['config']['variant']!=c['world_variant']
                or (world_provenance['config']['budget_override'] and not c['allow_world_smoke_budget'])):
            raise ValueError('World variant or completed training budget mismatch')
        policy_provenance_path = run/'provenance.json'
        policy_provenance = json.loads(policy_provenance_path.read_text())
        if (policy_provenance['config']!=c or policy_provenance['training_seed']!=260915
                or policy_provenance['protocol_sha256']!=sha(R/'protocol.json')
                or policy_provenance['independent_scorer_sha256']!=sha(SCORER)
                or policy_provenance['world_training_provenance_sha256']!=sha(provenance_path)):
            raise ValueError('World-PPO training provenance/config/scorer mismatch')
        policy_sources = [R/'ppo_world_worker.py',R/'train_world_policy.py',R/'ppo_wide_worker.py',
                          R/'physiologic_features.py',R/'world_control_worker.py',R/'protocol.json']
        bound_sources = policy_provenance['source_sha256']
        if not {str(p.relative_to(P)) for p in world_sources+policy_sources+[SCORER]}.issubset(bound_sources):
            raise ValueError('World-PPO provenance omits a required source dependency')
        for relative,digest in bound_sources.items():
            source=(P/relative).resolve(); source.relative_to(P)
            if sha(source)!=digest:
                raise ValueError('World-PPO training source hash mismatch: '+relative)
            sources.append(source)
        # The inherited wide worker imports these definitions but never runs its
        # D05/H02 initializer. Record the imported code without loading old weights.
        sources += world_sources+policy_sources+[B/'world_model.py',
            P/'RL_DITR创新_2026-09-16/ditr_model.py',P/'RL_DITR创新_2026-09-16/response_operator.py']
        artifacts += world_artifacts+[checkpoint,config,policy_provenance_path]
        command = [str(P/'.venv-native/bin/python'),'-u',str(R/'ppo_world_worker.py'),
                   '--config',str(config),'--checkpoint',str(checkpoint)]
    return (sorted(set(sources)), sorted(set(artifacts)), command, checkpoint)


class JsonWorker:
    """Bounded reads keep startup/process failures visible; stderr is preserved."""
    def __init__(self, command, stderr_path):
        self.log = stderr_path.open('wb')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log)
        except Exception:
            self.log.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.pending = b''

    def receive(self, timeout=60):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(max(0, remaining)):
                raise TimeoutError('JSON worker did not respond within %s seconds' % timeout)
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('JSON worker stdout closed; see worker.stderr.log')
            self.pending += chunk
        line, self.pending = self.pending.split(b'\n', 1)
        reply = json.loads(line)
        if not isinstance(reply, dict) or 'error' in reply:
            raise RuntimeError('JSON worker error: ' + repr(reply))
        return reply

    def request(self, payload):
        self.process.stdin.write((json.dumps(payload, allow_nan=False)+'\n').encode())
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        try:
            self.process.stdin.close()  # EOF exits both workers; no train/update operation.
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        self.selector.close(); self.process.stdout.close(); self.log.close()


def run(args):
    family=MEAN_KINDS.get(args.kind,args.kind)
    protocol = json.loads((R/'protocol.json').read_text())
    jobs = make_jobs(protocol, args.split, args.smoke)
    method=getattr(args,'method',None)
    if (args.kind=='retained' and method not in RETAINED_METHODS) or (args.kind!='retained' and method is not None):
        raise ValueError('--method is required only for --kind retained')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('--name must be one new output directory name')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    if family in ('ppo', 'world', 'world_ppo', 'iql_wide') and (args.checkpoint is None or args.config is None):
        raise ValueError('PPO/world/world-PPO/IQL evaluation requires --checkpoint and --config')
    if family not in ('ppo', 'world', 'world_ppo', 'iql_wide') and (args.checkpoint is not None or args.config is not None):
        raise ValueError('Only PPO/world/world-PPO/IQL accept overrides; legacy uses the frozen selected D06')
    if P/protocol['scorer'] != SCORER or sha(SCORER) != protocol['scorer_sha256']:
        raise ValueError('Original scorer path/hash mismatch')
    out = R/'results'/args.name
    out.mkdir(parents=True, exist_ok=False)
    (out/'trajectories').mkdir()
    started = time.time(); active = []; results = {}; worker = None; error = None
    retained_decisions={job_key(job):[] for job in jobs} if args.kind=='retained' else {}
    manifest = dict(schema=1, kind=args.kind, name=args.name, split=args.split, smoke=args.smoke,
                    method=method,
                    protocol=protocol, jobs=jobs, jobs_sha256=json_sha(jobs),
                    training_seed=260915, frozen_baseline=args.kind == 'legacy',
                    policy_input_fields=['history', 'observed_warmup_anchor'],
                    true_BG_reward_or_scoring_only=True, future_information_to_policy=False,
                    hidden_state_to_policy=False, tuning_split=True, confirmation=False,
                    action_support=('anchor +/- 0.25 U/h, clipped 0..20' if args.kind == 'legacy'
                                    else '0..min(20,2*observed anchor) U/h'),
                    action_projection=args.kind=='retained', scorer_sha256=sha(SCORER),
                    checkpoint_sha256=None,
                    source_sha256={str(p.relative_to(P)):sha(p) for p in COMMON_SOURCES},
                    artifact_sha256={str(NORMALIZER_PATH.relative_to(P)):sha(NORMALIZER_PATH),
                                     str((R/'protocol.json').relative_to(P)):sha(R/'protocol.json')})
    write_json(out/'manifest.json', manifest)
    provenance = dict(manifest_sha256=sha(out/'manifest.json'), jobs_sha256=manifest['jobs_sha256'],
                      scorer_sha256=manifest['scorer_sha256'], checkpoint_sha256=None,
                      source_sha256=manifest['source_sha256'])

    def save(job, raw=None, technical_error=None):
        key = job_key(job)
        if key in results:
            return
        if raw is None:
            raw = dict(job=job, records=[], failure_reason=technical_error,
                       raw_trajectory_available=False, unknown_tail=True)
        else:
            if raw['job'] != job:
                raise ValueError('Simulator returned a different job')
            raw['raw_trajectory_available'] = True
        if technical_error:
            raw['technical_failure_reason'] = technical_error
            raw['failure_reason'] = raw['failure_reason'] or technical_error
        planned = (job['total_minutes']-protocol['warmup_minutes'])//5
        metrics = summarize(raw['records'], planned, raw['failure_reason'] is not None)
        if 'metrics' in raw and not technical_error and raw['metrics'] != metrics:
            raise ValueError('Original worker/scorer result mismatch')
        if args.kind=='retained':
            decisions=retained_decisions[key]
            by_minute={d['control_interval_end_minute']:d for d in decisions}
            for row in raw['records']:
                if not row['warmup'] and row['minute'] in by_minute:
                    decision=by_minute[row['minute']]
                    row.update({k:decision[k] for k in ('raw_policy_u_h','observed_anchor_u_h','projected','projection_delta_u_h')})
            raw.update(retained_method=method,deployment_label='frozen+projection',
                       projection=projection_summary(decisions),
                       projection_delivered_link='Raw records join by control_interval_end_minute; delivered rate remains the original simulator field.')
        raw.update(metrics=metrics, evaluation_provenance=provenance,
                   observed_anchor_u_h=next((i['anchor'] for i in active if i['job']==job), None))
        path = out/'trajectories'/(key+'.json')
        write_json(path, raw)
        results[key] = dict(key=key, patient=job['patient'], seed=job['seed'], group=job['group'],
                            bolus_factor=job['bolus_factor'], failure_reason=raw['failure_reason'],
                            technical_failure_reason=raw.get('technical_failure_reason'),
                            raw_trajectory_available=raw['raw_trajectory_available'],
                            raw_path=str(path.relative_to(out)), raw_sha256=sha(path), metrics=metrics)
        if args.kind=='retained':
            results[key].update(method=method,deployment_label='frozen+projection',projection=raw['projection'])
        write_json(out/'progress.json', dict(recorded=len(results), total=len(jobs),
                                             elapsed_seconds=time.time()-started))
        print(json.dumps(dict(event='episode_recorded', key=key, count=len(results),
                              failure=raw['failure_reason'])), flush=True)

    try:
        if args.kind=='iql_wide' and (not args.checkpoint.is_absolute() or not args.config.is_absolute()):
            raise ValueError('IQL evaluation requires absolute checkpoint and run-config paths')
        checkpoint = args.checkpoint.resolve() if args.checkpoint else None
        config = args.config.resolve() if args.config else None
        sources, artifacts, command, checkpoint = dependencies(args.kind, checkpoint, config,method=method,
                                                              allow_smoke=args.smoke)
        manifest.update(source_sha256={str(p.relative_to(P)):sha(p) for p in sources},
                        artifact_sha256={str(p.relative_to(P)) if p.is_relative_to(P) else str(p):sha(p)
                                         for p in artifacts},
                        checkpoint=str(checkpoint) if checkpoint else None,
                        checkpoint_sha256=sha(checkpoint) if checkpoint else None,
                        config_sha256=sha(config) if config else None,
                        world_checkpoint_sha256=sha(checkpoint) if args.kind=='world' else None,
                        controller_family='MPC' if args.kind=='world' else args.kind,
                        worker_command=command, batch_size=args.batch_size)
        if config:
            manifest[family+'_config'] = json.loads(config.read_text())
        if family in ('ppo','world_ppo'):
            binding,_,_,prefix=policy_evaluation_binding(args.kind,checkpoint,config,args.smoke)
            snapshot=out/'policy_training_history_prefix.jsonl'; snapshot.write_bytes(prefix)
            manifest.update(policy_evaluation_binding=binding,
                            policy_training_history_snapshot=str(snapshot.relative_to(out)))
            manifest['artifact_sha256'][str(snapshot.relative_to(P))]=binding['history_prefix_sha256']
        if family in ('world','world_ppo'):
            wp=checkpoint if args.kind=='world' else Path(manifest['world_ppo_config']['world_checkpoint'])
            wrun=wp.parent.parent if wp.parent.name=='checkpoints' else wp.parent
            wpath=wrun/'provenance.json'
            manifest['world_evaluation_binding']=world_evaluation_binding(
                wpath,json.loads(wpath.read_text()),args.smoke)
        if family=='world_ppo':
            c=manifest['world_ppo_config']
            manifest.update(world_checkpoint=c['world_checkpoint'],world_checkpoint_sha256=c['world_checkpoint_sha256'],
                            policy_training_provenance_sha256=sha(config.parent/'provenance.json'),
                            controller_family='frozen_world_features_real_PPO')
        if args.kind in MEAN_KINDS:
            manifest.update(deployment_rule=MEAN_RULE,original_deployment_rule='argmax',
                underlying_policy_kind=family,same_checkpoint_new_deployment_ablation=True,
                proposed_after_ppo08_development_results=True,original_preregistration=False,
                development_observation_motivating_ablation='Root-reported PPO08: all 47520 deployed actions matched hold; mechanism hypothesis, not proof of useful probability changes.',
                new_training=False,new_external_method=False)
        if args.kind=='iql_wide':
            training=json.loads((config.parent/'provenance.json').read_text())
            manifest.update(controller_family='offline_IQL_new_wide_task',
                policy_training_provenance_sha256=sha(config.parent/'provenance.json'),
                policy_training_completion_sha256=sha(config.parent/'completion.json'),
                replay_manifest_sha256=training['replay_manifest_sha256'],
                iql_training_smoke=training['smoke'],checkpoint_selection='fixed final 20000; no development selection',
                action_parameterization='anchor*(normalized_mean+1), capped20; not retained absolute-action projection')
        if args.kind=='retained':
            retained,entry=retained_definition(method)
            config=R/'configs/retained_methods.json'
            manifest.update(method=method,method_label=entry['label'],deployment_label='frozen+projection',
                config_sha256=sha(config),retained_method=entry,retained_config=retained,
                frozen_baseline=True,action_projection=True,controller_family='retained_frozen_with_common_projection',
                policy_input_fields=['history'],projection_input_fields=['observed_warmup_anchor'],
                rng_transport_fields=[] if method=='ditr' else ['case_keys','scenario_seeds'],
                rng_fields_enter_network=False,raw_action_semantics='Original worker output in absolute U/h; no anchor rescaling',
                projection_definition=retained['action']['projection'])
        controller = make_controller(args.kind) if args.kind in ('hold', 'physiology') else None
        if controller is not None:
            manifest['controller_parameters'] = getattr(controller, 'parameters', {})
        for source in sources:
            target = out/'source'/source.relative_to(P)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        write_json(out/'manifest.json', manifest)
        provenance.update(manifest_sha256=sha(out/'manifest.json'), jobs_sha256=manifest['jobs_sha256'],
                          scorer_sha256=manifest['scorer_sha256'], checkpoint_sha256=manifest['checkpoint_sha256'],
                          config_sha256=manifest['config_sha256'],
                          world_checkpoint_sha256=manifest['world_checkpoint_sha256'],
                          source_sha256=manifest['source_sha256'])
        if family in ('world_ppo','iql_wide'):
            provenance['policy_training_provenance_sha256']=manifest['policy_training_provenance_sha256']
        if family in ('ppo','world_ppo'):
            provenance['policy_evaluation_binding']=manifest['policy_evaluation_binding']
        if family in ('world','world_ppo'):
            provenance['world_evaluation_binding']=manifest['world_evaluation_binding']
        if args.kind in MEAN_KINDS:
            provenance.update(deployment_rule=MEAN_RULE,underlying_policy_kind=family,
                same_checkpoint_new_deployment_ablation=True,original_preregistration=False,
                proposed_after_ppo08_development_results=True,new_training=False,new_external_method=False)
        if args.kind=='iql_wide':
            provenance.update(policy_training_completion_sha256=manifest['policy_training_completion_sha256'],
                              replay_manifest_sha256=manifest['replay_manifest_sha256'],
                              training_smoke=manifest['iql_training_smoke'])
        if command:
            worker = JsonWorker(command, out/'worker.stderr.log')
            ready = worker.receive()
            write_json(out/'worker_ready.json', ready)
            if not ready.get('ready'):
                raise RuntimeError('Worker failed readiness check')
            if family in ('ppo','world_ppo'):
                validate_policy_ready(ready,manifest['policy_evaluation_binding'],checkpoint,config)
            if args.kind in MEAN_KINDS:
                validate_mean_ready(ready,args.kind,manifest)
            if family in ('world','world_ppo'):
                loaded=ready['provenance'] if args.kind=='world' else ready['provenance']['world']
                if any(loaded.get(k)!=v for k,v in manifest['world_evaluation_binding'].items()):
                    raise ValueError('Worker world completed budget differs from the evaluation binding')
            if args.kind=='iql_wide':
                validate_iql_ready(ready,manifest)
            if args.kind=='world':
                bound=ready['provenance']
                if (bound['world_checkpoint_sha256']!=manifest['world_checkpoint_sha256']
                        or bound['control_config_sha256']!=manifest['config_sha256']):
                    raise ValueError('World worker loaded different checkpoint/config bytes')
            if family=='world_ppo':
                bound=ready['provenance']; c=manifest['world_ppo_config']
                if (ready.get('feature_dim')!=195 or ready.get('iteration')!=int(checkpoint.stem[len('policy_iter'):])
                        or bound['world']['world_checkpoint_sha256']!=c['world_checkpoint_sha256']
                        or bound['world']['training_provenance_sha256']!=c['world_training_provenance_sha256']
                        or bound['world']['completion_sha256']!=c['world_completion_sha256']
                        or bound['action_multipliers']!=c['action_multipliers']
                        or sha(checkpoint)!=manifest['checkpoint_sha256'] or sha(config)!=manifest['config_sha256']):
                    raise ValueError('World-PPO worker loaded different policy/config/world bytes or interface')
            provenance['worker_ready_sha256']=sha(out/'worker_ready.json')
        context = mp.get_context('spawn')
        with (out/'decisions.jsonl').open('w', buffering=1) as audit:
            for offset in range(0, len(jobs), args.batch_size):
                active = []
                for job_index,job in enumerate(jobs[offset:offset+args.batch_size],offset):
                    parent, child = context.Pipe()
                    process = context.Process(target=environment_worker, args=(child, job))
                    process.start(); child.close()
                    active.append(dict(pipe=parent, process=process, job=job, anchor=None, step=0,
                                       episode_handle=episode_handle(job_index)))
                while active:
                    pending = []; histories = []
                    for item in active:
                        if not item['pipe'].poll(60):
                            raise TimeoutError('Simulator did not respond: '+job_key(item['job']))
                        message = item['pipe'].recv()
                        if message['done']:
                            item['process'].join(5)
                            raw = message['result']; save(item['job'], raw)
                            item['pipe'].close()
                            if raw['failure_reason'] not in (None, 'native_environment_done', 'nonfinite_environment'):
                                raise RuntimeError('Technical simulator failure: '+str(raw['failure_reason']))
                        else:
                            h = np.asarray(message['history'], dtype=np.float32)
                            if h.shape != (72,22) or not np.isfinite(h).all():
                                raise ValueError('Invalid observed history')
                            if item['anchor'] is None:
                                item['anchor'] = warmup_anchor(h)
                            pending.append(item); histories.append(h)
                    active = pending
                    if not active:
                        break
                    h = np.stack(histories); anchors = np.array([i['anchor'] for i in active])
                    begin = time.perf_counter()
                    if controller is not None:
                        actions = controller.action(h, anchors)
                    else:
                        payload=policy_payload(args.kind,h,anchors,method=method,
                            episode_handles=[i['episode_handle'] for i in active],
                            scenario_seeds=[i['job']['seed'] for i in active])
                        response = worker.request(payload)
                        actions = response['actions'] if family in ('ppo','world','world_ppo','iql_wide') else response['actions_u_h']
                    if args.kind=='retained':
                        raw_actions=np.asarray(actions,dtype=np.float64)
                        actions,projected,projection_delta=project_retained_actions(raw_actions,anchors)
                    actions = validate_actions(args.kind, actions, anchors)
                    if args.kind in MEAN_KINDS:
                        mean_diagnostics=validate_mean_response(response,actions,anchors)
                    if args.kind=='iql_wide':
                        normalized,capped=validate_iql_response(response,actions,anchors)
                    if args.kind=='world':
                        from world_control_worker import candidate_plans
                        chosen=np.asarray(response['chosen_indices'])
                        if (chosen.shape!=anchors.shape or chosen.dtype.kind not in 'iu'
                                or np.any((chosen<0)|(chosen>=9)) or len(response['diagnostics'])!=len(active)):
                            raise ValueError('World worker returned invalid candidate diagnostics')
                        planned=candidate_plans(anchors.astype(np.float32))[np.arange(len(active)),chosen,0]
                        if not np.allclose(actions,planned,rtol=0,atol=1e-6):
                            raise ValueError('World action does not match the chosen first plan step')
                    if args.kind=='world_ppo':
                        chosen=np.asarray(response['action_indices'])
                        if chosen.shape!=anchors.shape or chosen.dtype.kind not in 'iu' or np.any((chosen<0)|(chosen>=9)):
                            raise ValueError('World-PPO returned invalid action indices')
                        grid=np.minimum(20,anchors.astype(np.float32)*(chosen.astype(np.float32)/4))
                        if not np.allclose(actions,grid,rtol=0,atol=1e-6):
                            raise ValueError('World-PPO action/index mismatch')
                    elapsed = time.perf_counter()-begin
                    for index,(item, action) in enumerate(zip(active, actions)):
                        decision=dict(key=job_key(item['job']), decision=item['step'],
                                      observed_anchor_u_h=item['anchor'], action_u_h=float(action),
                                      batch_size=len(active), batch_seconds=elapsed)
                        if args.kind=='world':
                            decision.update(chosen_index=int(chosen[index]),world_diagnostics=response['diagnostics'][index])
                        if args.kind=='world_ppo':
                            decision.update(chosen_index=int(chosen[index]))
                        if args.kind in MEAN_KINDS:
                            decision.update(deployment_rule=MEAN_RULE,**mean_diagnostics[index])
                        if args.kind=='iql_wide':
                            decision.update(normalized_action=float(normalized[index]),
                                            global_cap_applied=bool(capped[index]))
                        if args.kind=='retained':
                            decision.update(method=method,deployment_label='frozen+projection',
                                control_interval_end_minute=protocol['warmup_minutes']+5*(item['step']+1),
                                raw_policy_u_h=float(raw_actions[index]),requested_basal_u_h=float(action),
                                projected=bool(projected[index]),projection_delta_u_h=float(projection_delta[index]),
                                episode_rng_handle=item['episode_handle'] if method!='ditr' else None)
                            retained_decisions[job_key(item['job'])].append(decision)
                        audit.write(json.dumps(decision,allow_nan=False)+'\n')
                        item['pipe'].send(dict(action_u_h=float(action))); item['step'] += 1
                    if active[0]['step'] % 144 == 0:
                        print(json.dumps(dict(event='progress', offset=offset, active=len(active),
                                              step=active[0]['step'], seconds=time.time()-started)), flush=True)
    except Exception as caught:
        error = repr(caught)
        write_json(out/'failure.json', dict(error=error, traceback=traceback.format_exc(),
                                           worker_returncode=worker.process.poll() if worker else None))
        # Request genuine partial records from living simulators; never invent BG.
        for item in active:
            try:
                item['pipe'].send(dict(abort='Evaluation technical failure: '+error))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic()+10
        for item in active:
            if job_key(item['job']) in results:
                continue
            raw = None
            try:
                while item['pipe'].poll(max(0, deadline-time.monotonic())):
                    message = item['pipe'].recv()
                    if message['done']:
                        raw = message['result']; break
            except (EOFError, OSError):
                pass
            save(item['job'], raw, technical_error=error)
    finally:
        for item in active:
            if item['process'].is_alive():
                item['process'].terminate(); item['process'].join(5)
            item['pipe'].close()
        if worker is not None:
            try:
                worker.close()
                if worker.process.returncode != 0 and error is None:
                    error = 'JSON worker exited with code '+str(worker.process.returncode)
                    write_json(out/'failure.json', dict(error=error))
            except Exception as caught:
                if error is None:
                    error = 'Worker cleanup failed: '+repr(caught)
                    write_json(out/'failure.json', dict(error=error))
        for job in jobs:
            if job_key(job) not in results:
                save(job, technical_error='Not completed after technical failure: '+str(error))
        ordered = [results[job_key(j)] for j in jobs]
        retained_summary=dict(method=method,deployment_label='frozen+projection',
                              projection=projection_summary([d for rows in retained_decisions.values() for d in rows])) if args.kind=='retained' else {}
        write_json(out/'summary.json', dict(status='technical_failure' if error else 'completed',
                    error=error, kind=args.kind, split=args.split, smoke=args.smoke,
                    count=len(ordered), planned_count=len(jobs),
                    failed_episode_count=sum(r['metrics']['failed'] for r in ordered),
                    full_coverage_episode_count=sum(r['metrics']['bg']['coverage_pct']==100 and
                                                    r['metrics']['cgm']['coverage_pct']==100 for r in ordered),
                    episodes=ordered, provenance=provenance, jobs_sha256=manifest['jobs_sha256'],
                    wall_seconds=time.time()-started, training_seed=260915,
                    patient_claim=protocol['patient_claim'],**retained_summary))
    if error:
        raise RuntimeError('Evaluation failed; preserved evidence at '+str(out)+': '+error)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['physiology', 'hold', 'legacy', 'ppo', 'world', 'world_ppo', 'retained', 'iql_wide', 'ppo_mean', 'world_ppo_mean'], required=True)
    parser.add_argument('--method',choices=RETAINED_METHODS)
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', choices=['development', 'world_validation'], default='development')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=12)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
