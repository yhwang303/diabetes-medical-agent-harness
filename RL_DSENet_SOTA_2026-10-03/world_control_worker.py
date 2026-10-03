"""Frozen-world MPC mechanism ablations, not reinforcement learning.

The nine candidate plans are exactly paired_collect.plans: six hours at the
observed anchor, or 60/120 minutes at 0/.5/1.5/2 times anchor followed by anchor.
Only the first five minutes are executed; later requests replan from history.
Quantile expectation is midpoint-bin numerical quadrature, not a calibrated
distribution claim. Event outputs describe any BG threshold crossing over the
whole requested six hours; their penalties are not divided by 72 again.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
B = P/'RL_DSENet_2026-09-17'
NORMALIZER = P/'Loop数据集/训练管线_v2/prepared/normalization.json'
QUANTILES = [.05, .1, .25, .5, .75, .9, .95]
FROZEN_CONFIG = dict(schema=1, controller_family='MPC', horizon_steps=72,
                     control_interval_minutes=5, high_risk_weight=1., low_risk_weight=2.,
                     below70_event_weight=2., below54_event_weight=4., action_cost_weight=.02,
                     glucose_risk_clip_mg_dl=[20.,600.], torch_threads=2,
                     precision='float32_no_amp_no_tf32')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_control_config(path):
    config = json.loads(Path(path).read_text())
    mode = config.get('mode')
    if mode not in ('point', 'expected', 'risk') or config != dict(FROZEN_CONFIG, mode=mode):
        raise ValueError('Expected one of the three frozen MPC configurations')
    return config


def inspect_provenance(checkpoint):
    """Read only explicit training metadata/source; never load any label arrays."""
    checkpoint = Path(checkpoint).resolve()
    checkpoint.relative_to((R/'results').resolve())
    run = checkpoint.parent.parent if checkpoint.parent.name == 'checkpoints' else checkpoint.parent
    path = run/'provenance.json'
    provenance = json.loads(path.read_text())
    sources = provenance['source_sha256']
    required = [R/'train_world_v2.py', R/'world_model_v2.py', R/'protocol.json',
                R/'prepare_windows.py', R/'paired_collect.py', B/'forecast_model.py']
    required += sorted((B/'dsenet').rglob('*.py'))
    if not set(str(p.relative_to(P)) for p in required).issubset(sources):
        raise ValueError('Training provenance omits a required source dependency')
    paths = []
    for relative, digest in sources.items():
        source = (P/relative).resolve()
        source.relative_to(P)
        if sha(source) != digest:
            raise ValueError('Training source hash mismatch: '+relative)
        paths.append(source)
    if provenance['protocol_sha256'] != sha(R/'protocol.json') or provenance['seed'] != 260915:
        raise ValueError('Training protocol/seed binding mismatch')
    for kind in ('forecast', 'normalization'):
        binding = provenance[kind]
        if not Path(binding['path']).is_absolute() or sha(binding['path']) != binding['sha256']:
            raise ValueError('Frozen '+kind+' dependency hash mismatch')
        if provenance['config'][kind] != binding:
            raise ValueError('Training config/provenance '+kind+' mismatch')
    if Path(provenance['normalization']['path']).resolve() != NORMALIZER.resolve():
        raise ValueError('Inference must use the existing Loop normalization')
    if provenance['normalization']['sha256'] != sha(NORMALIZER):
        raise ValueError('Inference Loop normalization hash mismatch')
    selected_path = R/'checks/frozen_selected_version.json'
    selected = json.loads(selected_path.read_text())['weights']['forecast']
    if (Path(provenance['forecast']['path']).resolve() != (B/selected['path']).resolve()
            or provenance['forecast']['sha256'] != selected['sha256']):
        raise ValueError('World model is not bound to the frozen P03 forecast')
    completion_path = run/'completion.json'
    completion = json.loads(completion_path.read_text())
    config = provenance['config']
    if (completion['status']!='training_completed_candidate_only'
            or completion['steps']!=config['steps'] or completion['configured_steps']!=config['configured_steps']
            or completion['budget_override']!=config['budget_override']
            or completion['selected_on']!='world_validation'
            or sha(checkpoint) not in (completion['best_checkpoint_sha256'],completion['last_checkpoint_sha256'])):
        raise ValueError('Only bound best/last checkpoints from a completed training budget are accepted')
    artifacts = [checkpoint, path, completion_path, NORMALIZER, Path(provenance['forecast']['path']), selected_path]
    return path, provenance, paths, artifacts


def candidate_plans(anchors):
    # Reuse the actual intervention implementation, including its 0..20 clipping.
    from paired_collect import plans
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.ndim != 1 or len(anchors) == 0 or not np.isfinite(anchors).all() or np.any((anchors<=0)|(anchors>20)):
        raise ValueError('Expected positive finite observed anchors <=20 U/h')
    return np.stack([plans(float(a)) for a in anchors]).astype(np.float32)


def quantile_weights(levels):
    levels = np.asarray(levels, dtype=np.float64)
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or np.any(np.diff(levels)<=0) or levels[0]<=0 or levels[-1]>=1):
        raise ValueError('Invalid ordered quantile levels')
    boundaries = np.r_[0., (levels[:-1]+levels[1:])/2., 1.]
    return np.diff(boundaries)


def weighted_risk(glucose):
    """High risk weight 1, low risk weight 2 after explicit 20..600 clipping."""
    glucose = np.asarray(glucose, dtype=np.float64)
    if not np.isfinite(glucose).all():
        raise ValueError('Nonfinite predicted glucose is a technical failure')
    f = 1.509*(np.log(np.clip(glucose,20,600))**1.084-5.381)
    risk = 10*f*f
    return np.where(f<0, 2*risk, risk)


def score_predictions(quantiles, levels, p70, p54, plans, anchors, mode):
    """Pure numeric scoring; arguments contain predictions/actions, never labels."""
    q = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    plans = np.asarray(plans, dtype=np.float64); anchors = np.asarray(anchors, dtype=np.float64)
    p70 = np.asarray(p70, dtype=np.float64); p54 = np.asarray(p54, dtype=np.float64)
    if (q.shape != (len(anchors),9,72,len(levels)) or plans.shape != q.shape[:3]
            or p70.shape != q.shape[:2] or p54.shape != p70.shape):
        raise ValueError('World scoring expects N x 9 x 72 predictions')
    if mode not in ('point','expected','risk'):
        raise ValueError('Unknown MPC ablation')
    weights = quantile_weights(levels)
    median = np.flatnonzero(np.isclose(levels,.5,rtol=0,atol=1e-7))
    if len(median)!=1 or (mode != 'point' and (len(levels)!=len(QUANTILES)
                                              or not np.allclose(levels,QUANTILES,rtol=0,atol=1e-7))):
        raise ValueError('Expected median, and seven quantiles for probabilistic modes')
    if not np.isfinite(q).all() or np.any(np.diff(q,axis=-1)<0):
        raise ValueError('Nonfinite or crossing predicted quantiles')
    if (not np.isfinite(p70).all() or not np.isfinite(p54).all()
            or np.any((p54<0)|(p54>p70+1e-7)|(p70>1+1e-7))):
        raise ValueError('Invalid ordered BG event probabilities')
    if not np.isfinite(plans).all() or not np.isfinite(anchors).all() or np.any(anchors<=0):
        raise ValueError('Invalid plan/anchor')
    used = q[...,median[0]:median[0]+1] if mode=='point' else q
    trajectory_risk = weighted_risk(used)
    if mode=='point':
        glucose_cost = trajectory_risk[...,0].mean(-1)
    else:
        glucose_cost = (trajectory_risk*weights).sum(-1).mean(-1)
    event_cost = 2*p70+4*p54 if mode=='risk' else np.zeros_like(p70)
    action_cost = .02*np.square(plans/anchors[:,None,None]-1).mean(-1)
    total = glucose_cost+event_cost+action_cost
    if not np.isfinite(total).all():
        raise ValueError('Nonfinite MPC score')
    chosen = total.argmin(-1)  # NumPy keeps first index, hence hold on exact ties.
    clipped = ((used<20)|(used>600)).mean((-1,-2))
    diagnostics = [dict(candidate_total_cost=total[i].tolist(),
                        candidate_glucose_risk=glucose_cost[i].tolist(),
                        candidate_event_cost=event_cost[i].tolist(),
                        candidate_action_cost=action_cost[i].tolist(),
                        candidate_p_bg_below70=p70[i].tolist(),candidate_p_bg_below54=p54[i].tolist(),
                        candidate_risk_input_clip_fraction=clipped[i].tolist(),
                        selected_risk_input_clip_fraction=float(clipped[i,chosen[i]]))
                   for i in range(len(anchors))]
    return plans[np.arange(len(anchors)),chosen,0], chosen, diagnostics


def load_world(checkpoint_path):
    """Return a strictly bound frozen CUDA model and its provenance summary.

    A completed explicit smoke-budget model is loadable for engineering checks;
    `budget_override` remains explicit so a formal PPO caller can reject it.
    No training data arrays are read, and no optimizer/checkpoint is written.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        raise ValueError('World checkpoint must be an absolute path')
    provenance_path, provenance, sources, _ = inspect_provenance(checkpoint)
    completion = json.loads((provenance_path.parent/'completion.json').read_text())
    import torch
    from world_model_v2 import WorldModelV2, load_frozen_dsenet
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/Mamba runtime required; no CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    state = torch.load(checkpoint, map_location='cpu')
    if (state['format_version']!=1 or state['model_type']!='WorldModelV2'
            or state['training_seed']!=260915 or state['provenance_sha256']!=sha(provenance_path)
            or state['config']!=provenance['config']):
        raise ValueError('World checkpoint/training provenance mismatch')
    for kind in ('forecast','normalization'):
        if state[kind]!=provenance[kind]:
            raise ValueError('Checkpoint '+kind+' binding mismatch')
    training = state['config']; levels = state['quantiles']
    if state['width']!=training['width'] or levels!=training['quantiles']:
        raise ValueError('Checkpoint architecture/config mismatch')
    if levels not in ([.5],QUANTILES) or training['variant']!=('point' if levels==[.5] else 'quantile'):
        raise ValueError('Unknown point/quantile training variant')
    if training['precision']!='float32_no_amp_no_tf32':
        raise ValueError('Unsupported world training precision')
    forecast = load_frozen_dsenet(Path(state['forecast']['path']))
    norm = json.loads(Path(state['normalization']['path']).read_text())
    model = WorldModelV2(forecast, norm, width=state['width'], quantiles=levels)
    # Verify embedded frozen tensors as well as their original dependency files.
    for name,value in model.state_dict().items():
        if name.startswith('forecast.') or name in ('history_mean','history_scale','quantile_levels','exposure_weights'):
            if not torch.equal(value,state['model'][name]):
                raise ValueError('Embedded frozen tensor mismatch: '+name)
    model.load_state_dict(state['model'],strict=True)
    model = model.cuda().eval().requires_grad_(False)
    summary = dict(world_checkpoint_sha256=sha(checkpoint),training_provenance_sha256=sha(provenance_path),
        completion_sha256=sha(provenance_path.parent/'completion.json'),
        forecast_sha256=state['forecast']['sha256'],normalization_sha256=state['normalization']['sha256'],
        training_step=state['step'],training_variant=training['variant'],quantile_levels=levels,
        completed_steps=completion['steps'],configured_steps=completion['configured_steps'],
        budget_override=completion['budget_override'],
        quantile_integration_weights=quantile_weights(levels).tolist(),
        source_sha256={str(p.relative_to(P)):sha(p) for p in sources+[Path(__file__).resolve()]},
        action_plan_source_sha256=sha(R/'paired_collect.py'),horizon_steps=72,
        training=False,calibrated_probability_claim=False,
        matmul_allow_tf32=False,cudnn_allow_tf32=False,cudnn_deterministic=False,torch_threads=2)
    return model, summary


class WorldController:
    def __init__(self, checkpoint, config_path):
        if not config_path.is_absolute():
            raise ValueError('Control config path must be absolute')
        self.config = load_control_config(config_path)
        self.model, summary = load_world(checkpoint)
        if self.config['mode']!='point' and summary['quantile_levels']!=QUANTILES:
            raise ValueError('Expected/risk modes require the seven-quantile model')
        import torch
        self.torch = torch
        self.provenance = dict(summary,controller_family='MPC',mode=self.config['mode'],
                               control_config_sha256=sha(config_path),execute_first_step_only=True)

    def evaluate(self, request):
        if set(request)!={'op','history','anchors'} or request['op']!='evaluate':
            raise ValueError('Only evaluate/history/anchors are accepted; labels and hidden fields are forbidden')
        history = np.asarray(request['history'],dtype=np.float32)
        anchors = np.asarray(request['anchors'],dtype=np.float32)
        if history.shape != (len(anchors),72,22) or not np.isfinite(history).all():
            raise ValueError('Expected finite observed history[N,72,22]')
        plans = candidate_plans(anchors)
        torch = self.torch; begin=time.perf_counter()
        with torch.inference_mode():
            h=torch.as_tensor(history,device='cuda'); a=torch.as_tensor(anchors,device='cuda')
            output=self.model.predict(self.model.encode(h,a),torch.as_tensor(plans,device='cuda'))
            prediction={name:output[name].cpu().numpy() for name in
                        ('cgm_quantiles_mgdl','quantile_levels','p_bg_below70','p_bg_below54')}
        actions,chosen,diagnostics=score_predictions(prediction['cgm_quantiles_mgdl'],prediction['quantile_levels'],
            prediction['p_bg_below70'],prediction['p_bg_below54'],plans,anchors,self.config['mode'])
        return dict(actions=actions.tolist(),chosen_indices=chosen.tolist(),diagnostics=diagnostics,
                    batch_seconds=time.perf_counter()-begin)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); controller=WorldController(args.checkpoint,args.config)
    print(json.dumps(dict(ready=True,provenance=controller.provenance),allow_nan=False),flush=True)
    for line in sys.stdin:
        try:
            reply=controller.evaluate(json.loads(line))
        except Exception as error:
            reply=dict(error=repr(error))
        print(json.dumps(reply,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
