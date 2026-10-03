r"""Train WorldModelV2 on audited train data; select only on world_validation.

Run from this research directory, using cache names created by prepare_windows
and paired_collect (replace names with actual completed datasets):
  python train_world_v2.py --config configs/world_v2_quantile.json \
    --name world_quantile_run1 --natural-train cache/natural_train_windows_r1 \
    --paired-train cache/paired_train_r1 \
    --natural-validation cache/natural_validation_windows_r1 \
    --paired-validation cache/paired_world_validation_r1 \
    --forecast ../RL_DSENet_2026-09-17/results/P03_g12s3_l2s1/best.pt \
    --normalization ../Loop数据集/训练管线_v2/prepared/normalization.json
Use world_v2_point.json for the matched point ablation. --steps 4 creates a
smoke-budget run, which must use a new --name. No existing run is overwritten.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback


R = Path(__file__).resolve().parent
P = R.parent
ARRAY_KEYS = ('history', 'anchor_u_h', 'actions_u_h', 'target_cgm_mg_dl', 'target_bg_mg_dl', 'mask')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def inspect_manifest(directory, expected_split, paired, protocol):
    """Check split/scenario metadata BEFORE opening any NPZ arrays."""
    directory = directory.resolve()
    directory.relative_to((R / 'cache').resolve())
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'completed' or manifest.get('split') != expected_split:
        raise ValueError('Only completed ' + expected_split + ' datasets are permitted: ' + str(directory))
    if manifest.get('horizon') != 72:
        raise ValueError('Expected frozen 72-step future windows')
    if paired != (manifest.get('kind') == 'paired_interventions'):
        raise ValueError('Natural/paired dataset kind mismatch')
    allowed = set(protocol[expected_split + '_scenario_seeds'])
    if expected_split == 'train':
        lo, hi = protocol['ppo_train_scenario_seed_range_inclusive']
        allowed.update(range(lo, hi + 1))
    records = sorted(manifest['episodes'], key=lambda item: item['file'])
    if not records or len({item['file'] for item in records}) != len(records):
        raise ValueError('Empty dataset or repeated file in manifest')
    for item in records:
        job = item['job']
        if job['seed'] not in allowed or job['patient'] not in protocol['patients']:
            raise ValueError('Scenario/patient outside permitted split before reading arrays')
        if job['bolus_factor'] not in protocol['bolus_factors']:
            raise ValueError('Unexpected scenario bolus factor')
        name = Path(item['file'])
        if name.name != str(name) or name.suffix != '.npz':
            raise ValueError('Dataset manifest must name local NPZ files')
        (directory / name).resolve().relative_to(directory)
    return directory, manifest, records, sha(manifest_path)


def load_data(inspection, paired):
    directory, manifest, records, manifest_hash = inspection
    chunks = {key: [] for key in ARRAY_KEYS}
    order = []
    for record in records:
        path = directory / record['file']
        digest = sha(path)
        if digest != record['sha256']:
            raise ValueError('Dataset hash mismatch: ' + str(path))
        with np.load(path, allow_pickle=False) as packed:
            arrays = {key: packed[key] for key in ARRAY_KEYS}
        count = len(arrays['history'])
        expected_count = record['groups' if paired else 'origins']
        if count != expected_count or count < 1 or arrays['history'].shape != (count, 72, 22):
            raise ValueError('History/group count mismatch: ' + str(path))
        expected = (count, 9, 72) if paired else (count, 72)
        if arrays['anchor_u_h'].shape != (count,):
            raise ValueError('Anchor shape mismatch')
        for key in ARRAY_KEYS:
            value = arrays[key]
            if value.dtype != (np.bool_ if key == 'mask' else np.float32):
                raise ValueError('Unexpected dtype for ' + key)
            if key not in ('history', 'anchor_u_h') and value.shape != expected:
                raise ValueError('Future shape mismatch for ' + key)
            if key != 'mask' and not np.isfinite(value).all():
                raise ValueError('Nonfinite stored values for ' + key)
        mask = arrays['mask']
        if not mask.any(-1).all() or (mask[..., 1:] & ~mask[..., :-1]).any():
            raise ValueError('Expected nonempty observed prefixes; padded tails remain unknown')
        if (arrays['actions_u_h'] < 0).any() or (arrays['anchor_u_h'] < 0).any():
            raise ValueError('Negative action/anchor')
        for key in ARRAY_KEYS:
            value = arrays[key]
            if not paired and key not in ('history', 'anchor_u_h'):
                value = value[:, None]
            chunks[key].append(value)
        order.append(dict(file=record['file'],count=count,sha256=digest,job=record['job']))
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    full = arrays['mask'].all(-1)
    observed_minimum = np.where(arrays['mask'], arrays['target_bg_mg_dl'], np.inf).min(-1)
    provenance = dict(path=str(directory),manifest_sha256=manifest_hash,split=manifest['split'],
                      paired=paired,groups=len(arrays['history']),arms=arrays['mask'].shape[1],
                      full_windows=int(full.sum()),censored_windows=int((~full).sum()),
                      observed_low54_windows=int((observed_minimum<54).sum()),
                      full_low54_windows=int(((observed_minimum<54)&full).sum()),
                      file_order=order)
    return arrays, provenance


def to_batch(data, indices, device):
    return {key: torch.as_tensor(value[indices], device=device) for key, value in data.items()}


def loss_components(output, batch, paired):
    q = output['cgm_quantiles_mgdl']
    mask = batch['mask']
    error = batch['target_cgm_mg_dl'][..., None] - q
    levels = output['quantile_levels']
    pinball = torch.maximum(levels * error, (levels - 1) * error)
    per_arm = (pinball * mask[..., None]).sum((-1, -2)) / (mask.sum(-1) * len(levels))
    pinball_loss = per_arm.mean(1).mean()
    full = mask.all(-1)
    minimum = torch.where(mask, batch['target_bg_mg_dl'], float('inf')).amin(-1)
    labels = torch.where(minimum < 54, 2, torch.where(minimum < 70, 1, 0)).long()
    ce = torch.nn.functional.cross_entropy(output['bg_event_logits'].reshape(-1, 3),
                                          labels.reshape(-1), reduction='none').reshape_as(full)
    event_groups = full.any(1)
    group_ce = (ce * full).sum(1) / full.sum(1).clamp_min(1)
    event_loss = (group_ce * event_groups).sum() / event_groups.sum().clamp_min(1)
    pair_loss = q.sum() * 0.
    if paired:
        median = output['cgm_median_mgdl']
        difference = median[:, 1:] - median[:, :1]
        target = batch['target_cgm_mg_dl'][:, 1:] - batch['target_cgm_mg_dl'][:, :1]
        common = mask[:, 1:] & mask[:, :1]
        per_arm_delta = ((difference - target).abs() * common).sum(-1) / common.sum(-1).clamp_min(1)
        pair_loss = per_arm_delta.mean(1).mean()
    return dict(pinball=pinball_loss,event_ce=event_loss,paired_delta_mae=pair_loss), int(event_groups.sum())


def composite(natural, paired, config):
    return (.5 * (natural['pinball'] + paired['pinball']) / config['glucose_loss_scale_mg_dl']
            + .5 * config['event_weight'] * (natural['event_ce'] + paired['event_ce'])
            + config['paired_delta_weight'] * paired['paired_delta_mae'] / config['glucose_loss_scale_mg_dl'])


def ratio(value, count):
    return float(value / count) if count else None


def order_counts(prediction, target, tolerance=1e-6):
    """All upper-triangle arm pairs; omit ties in the observed quantity only."""
    i, j = np.triu_indices(prediction.shape[1], 1)
    true_difference = target[:, i] - target[:, j]
    predicted_difference = prediction[:, i] - prediction[:, j]
    eligible = np.abs(true_difference) > tolerance
    correct = (np.sign(true_difference) == np.sign(predicted_difference)) & eligible
    return int(correct.sum()), int(eligible.sum())


def validate_dataset(world, data, paired, out, step, config, device):
    count = len(data['history'])
    sums = dict(pinball=0.,event_ce=0.,paired_delta_mae=0.)
    event_groups = 0
    all_q, all_prob = [], []
    for begin in range(0, count, config['validation_batch_groups']):
        indices = slice(begin, min(count, begin + config['validation_batch_groups']))
        batch = to_batch(data, indices, device)
        output = world(batch['history'], batch['anchor_u_h'], batch['actions_u_h'])
        losses, valid_event_groups = loss_components(output, batch, paired)
        size = len(batch['history'])
        for key in sums:
            sums[key] += float(losses[key]) * (valid_event_groups if key == 'event_ce' else size)
        event_groups += valid_event_groups
        all_q.append(output['cgm_quantiles_mgdl'].cpu().numpy())
        all_prob.append(output['bg_event_probabilities'].cpu().numpy())
    components = {key: value / (max(1,event_groups) if key == 'event_ce' else count) for key,value in sums.items()}
    quantiles, probability = np.concatenate(all_q), np.concatenate(all_prob)
    levels = world.quantile_levels.cpu().numpy()
    median = quantiles[..., world.median_index]
    mask = data['mask']
    target = data['target_cgm_mg_dl']
    error = median - target
    complete = mask.all(-1)
    result = dict(groups=count,arms=mask.shape[1],valid_cgm_points=int(mask.sum()),
                  complete_event_windows=int(complete.sum()),event_supervised_groups=event_groups,
                  censored_event_windows_excluded=int((~complete).sum()),loss_components=components,
                  cgm_mae_mg_dl=float(np.abs(error)[mask].mean()),
                  cgm_rmse_mg_dl=float(np.sqrt(np.square(error)[mask].mean())),
                  diagnostic_scope='overlapping windows, not independent patients or closed-loop policy scores')
    np.savez_compressed(out / 'validation' / ('%s_step%07d.npz'%('paired' if paired else 'natural',step)),
                        cgm_quantiles_mgdl=quantiles,bg_event_probabilities=probability,
                        quantile_levels=levels,sample_index=np.arange(count))
    if not paired:
        result['quantile_coverage'] = {str(float(tau)):float((target<=quantiles[...,k])[mask].mean())
                                       for k,tau in enumerate(levels)}
        intervals = {}
        for lower,upper in ((.05,.95),(.1,.9),(.25,.75)):
            lo = np.flatnonzero(np.isclose(levels,lower));hi=np.flatnonzero(np.isclose(levels,upper))
            if len(lo) and len(hi):
                left,right = quantiles[...,lo[0]],quantiles[...,hi[0]]
                intervals[str(upper-lower)] = dict(coverage=float(((target>=left)&(target<=right))[mask].mean()),
                                                   mean_width_mg_dl=float((right-left)[mask].mean()))
        result['central_intervals'] = intervals
        result['bg_horizon_event_calibration'] = {}
        minimum = np.where(mask,data['target_bg_mg_dl'],np.inf).min(-1)
        for threshold,predicted in ((70,probability[...,1:].sum(-1)),(54,probability[...,2])):
            actual = (minimum[complete] < threshold).astype(np.float64)
            p = predicted[complete]
            positive = int(actual.sum());negative=len(actual)-positive
            bins=[]
            for k in range(10):
                take = (p>=k/10) & ((p<(k+1)/10) if k<9 else (p<=1))
                bins.append(dict(lower=k/10,upper=(k+1)/10,count=int(take.sum()),
                                 predicted_mean=float(p[take].mean()) if take.any() else None,
                                 observed_rate=float(actual[take].mean()) if take.any() else None))
            result['bg_horizon_event_calibration'][str(threshold)] = dict(
                count=len(actual),positive=positive,negative=negative,
                brier=float(np.square(p-actual).mean()) if len(actual) else None,
                predicted_mean=float(p.mean()) if len(p) else None,
                observed_rate=ratio(positive,len(actual)),
                fnr_at_probability_0_5=ratio(((p<.5)&(actual==1)).sum(),positive),
                fpr_at_probability_0_5=ratio(((p>=.5)&(actual==0)).sum(),negative),bins=bins)
        result['observed_bg_low_fraction_in_window_labels'] = {
            str(t):float((data['target_bg_mg_dl'][mask]<t).mean()) for t in (70,54)}
    else:
        full_groups = complete.all(1)
        result['paired_delta_mae_mg_dl'] = components['paired_delta_mae']
        result['ranking_complete_groups'] = int(full_groups.sum())
        cgm_correct,cgm_pairs = order_counts(median[full_groups].min(-1),target[full_groups].min(-1))
        result['minimum_cgm_order_accuracy'] = dict(accuracy=ratio(cgm_correct,cgm_pairs),pairs=cgm_pairs)
        # Event ranking uses actual complete-window event labels, omitting ties.
        bg_minimum=data['target_bg_mg_dl'][full_groups].min(-1)
        for threshold,predicted in ((70,probability[full_groups,:,1:].sum(-1)),(54,probability[full_groups,:,2])):
            correct,pairs = order_counts(predicted,(bg_minimum<threshold).astype(np.float32))
            result['bg_event_%d_order_accuracy'%threshold] = dict(accuracy=ratio(correct,pairs),pairs=pairs)
    return result


def matched_initial_model(forecast, normalization, config):
    # Shared layers and median row match exactly across the two architectures.
    canonical = WorldModelV2(forecast, normalization, width=config['width'])
    if config['variant'] == 'quantile':
        return canonical
    rng_after_canonical = torch.get_rng_state()
    point = WorldModelV2(forecast, normalization, width=config['width'], quantiles=(.5,))
    canonical_state = canonical.state_dict()
    with torch.no_grad():
        for name,value in point.state_dict().items():
            if name in ('quantile_levels','quantile_head.weight','quantile_head.bias'):
                continue
            value.copy_(canonical_state[name])
        row=canonical.median_index
        point.quantile_head.weight.copy_(canonical.quantile_head.weight[row:row+1])
        point.quantile_head.bias.copy_(canonical.quantile_head.bias[row:row+1])
    torch.set_rng_state(rng_after_canonical)
    return point


def parameter_digest(world, shared_only=False):
    digest=hashlib.sha256()
    for name,value in world.named_parameters():
        if shared_only and name.startswith('quantile_head.'):
            continue
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen_state_digest(forecast):
    digest=hashlib.sha256()
    for name,value in forecast.state_dict().items():
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, world, optimizer, rng, step, config, validation, provenance_hash):
    torch.save(dict(format_version=1,model_type='WorldModelV2',model=world.state_dict(),
                    width=config['width'],quantiles=config['quantiles'],
                    forecast=config['forecast'],normalization=config['normalization'],
                    optimizer=optimizer.state_dict(),step=step,config=config,
                    validation=validation,provenance_sha256=provenance_hash,training_seed=260915,
                    numpy_rng_state=rng.bit_generator.state,python_rng_state=random.getstate(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),path)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--natural-train',required=True);ap.add_argument('--paired-train',required=True)
    ap.add_argument('--natural-validation',required=True);ap.add_argument('--paired-validation',required=True)
    ap.add_argument('--forecast',required=True);ap.add_argument('--normalization',required=True)
    ap.add_argument('--steps',type=int);ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    if Path(args.name).name != args.name or args.name in ('','.','..'):
        raise ValueError('name must be a new simple directory name')
    out=R/'results'/args.name;out.mkdir(parents=True,exist_ok=False)
    begin=time.time();completed_steps=0
    try:
        for name in ('source','checkpoints','validation'):(out/name).mkdir()
        config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
        protocol=json.loads((R/'protocol.json').read_text())
        if config['seed'] != 260915 or config['seed'] != protocol['training_seed']:
            raise ValueError('Only training seed 260915 is authorized')
        if config['variant'] not in ('point','quantile'):
            raise ValueError('Expected point or quantile configuration')
        expected_q=[.5] if config['variant']=='point' else [.05,.1,.25,.5,.75,.9,.95]
        if config['quantiles'] != expected_q:
            raise ValueError('Quantile levels disagree with the frozen ablation variant')
        if config['natural_fraction'] != .5 or config['batch_groups']%2:
            raise ValueError('Use an even batch with 50 percent natural and paired groups')
        if config['precision'] != 'float32_no_amp_no_tf32':
            raise ValueError('This trainer implements the frozen FP32 configuration only')
        config=dict(config,configured_steps=config['steps'],steps=args.steps if args.steps is not None else config['steps'],
                    budget_override=args.steps is not None,run_name=args.name,device=args.device)
        if min(config['steps'],config['batch_groups'],config['validation_every'],config['validation_batch_groups'])<1:
            raise ValueError('Training and validation budgets must be positive')
        write_json(out/'config.json',config)
        shutil.copyfile(config_path,out/'source'/'requested_config.json')
        requested={
            'natural_train':(args.natural_train,'train',False),
            'paired_train':(args.paired_train,'train',True),
            'natural_validation':(args.natural_validation,'world_validation',False),
            'paired_validation':(args.paired_validation,'world_validation',True),
        }
        inspections={key:inspect_manifest(Path(path),split,paired,protocol)
                     for key,(path,split,paired) in requested.items()}
        # Import heavy runtimes only after split metadata has passed the gate.
        global np,torch,WorldModelV2,load_frozen_dsenet
        import numpy as np
        import torch
        from world_model_v2 import WorldModelV2,load_frozen_dsenet
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; do not silently replace the legacy DSENet runtime')
        random.seed(260915);np.random.seed(260915);torch.manual_seed(260915)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(260915)
        rng=np.random.default_rng(260915)
        torch.set_num_threads(config['torch_threads'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        datasets={};data_provenance={}
        for key,inspection in inspections.items():
            datasets[key],data_provenance[key]=load_data(inspection,requested[key][2])
            if not data_provenance[key]['full_windows']:
                raise ValueError('Dataset has no fully observed event window: '+key)
        normalization_path=Path(args.normalization).resolve();forecast_path=Path(args.forecast).resolve()
        normalization=json.loads(normalization_path.read_text())
        shutil.copyfile(normalization_path,out/'source'/'normalization.json')
        config.update(forecast=dict(path=str(forecast_path),sha256=sha(forecast_path),frozen=True),
                      normalization=dict(path=str(normalization_path),sha256=sha(normalization_path)))
        if config['forecast']['sha256'] != config['forecast_sha256']:
            raise ValueError('Frozen P03 checkpoint SHA mismatch before model loading')
        if config['normalization']['sha256'] != config['normalization_sha256']:
            raise ValueError('Frozen normalization SHA mismatch')
        write_json(out/'config.json',config)
        forecast=load_frozen_dsenet(forecast_path)
        for attr in ('mean','scale'):
            expected=torch.tensor(normalization['cgm_mmol_l'][attr],dtype=getattr(forecast,attr).dtype)
            if not torch.equal(getattr(forecast,attr),expected):
                raise ValueError('New history normalization differs from frozen DSENet')
        world=matched_initial_model(forecast,normalization,config).to(args.device)
        frozen_hash=frozen_state_digest(world.forecast)
        optimizer=torch.optim.AdamW([p for p in world.parameters() if p.requires_grad],
                                   lr=config['learning_rate'],weight_decay=config['weight_decay'])
        sources=[Path(__file__).resolve(),R/'world_model_v2.py',R/'protocol.json',R/'prepare_windows.py',R/'paired_collect.py',
                 P/'RL_DSENet_2026-09-17/forecast_model.py']
        sources+=sorted((P/'RL_DSENet_2026-09-17/dsenet').rglob('*.py'))
        source_hashes={}
        for path in sources:
            relative=path.relative_to(P);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);source_hashes[str(relative)]=sha(path)
        provenance=dict(config=config,requested_config_sha256=sha(config_path),source_sha256=source_hashes,
                        protocol_sha256=sha(R/'protocol.json'),data=data_provenance,
                        forecast=config['forecast'],normalization=config['normalization'],
                        runtime=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                                     cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                                     amp=False,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                     cudnn_deterministic=torch.backends.cudnn.deterministic,
                                     device=args.device,gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None),
                        initial_shared_parameter_sha256=parameter_digest(world,True),
                        initial_all_parameter_sha256=parameter_digest(world),
                        initial_frozen_forecast_state_sha256=frozen_hash,
                        trainable_parameters=sum(p.numel() for p in world.parameters() if p.requires_grad),
                        frozen_parameters=sum(p.numel() for p in world.parameters() if not p.requires_grad),
                        seed=260915,low_glucose_resampling=False,forecast_precomputed=False,
                        training_cache='rebuild trainable encode features every optimizer step',
                        calibrated_probability_claim=False,bitwise_reproducibility_not_assumed=True,
                        selection_scope='checkpoint selection within variant; scores not comparable between point and quantile objectives')
        write_json(out/'provenance.json',provenance);provenance_hash=sha(out/'provenance.json')
        save_checkpoint(out/'checkpoints'/'initial.pt',world,optimizer,rng,0,config,None,provenance_hash)
        best=math.inf;best_step=None;half=config['batch_groups']//2
        with (out/'history.jsonl').open('w',buffering=1) as log:
            for step in range(1,config['steps']+1):
                world.train();optimizer.zero_grad(set_to_none=True)
                sample_indices={};losses={};event_counts={}
                for kind in ('natural','paired'):
                    data=datasets[kind+'_train'];indices=rng.integers(0,len(data['history']),size=half)
                    sample_indices[kind]=indices.tolist();batch=to_batch(data,indices,args.device)
                    prediction=world(batch['history'],batch['anchor_u_h'],batch['actions_u_h'])
                    losses[kind],event_counts[kind]=loss_components(prediction,batch,kind=='paired')
                loss=composite(losses['natural'],losses['paired'],config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in world.parameters() if p.requires_grad],
                                                   config['gradient_clip'],error_if_nonfinite=True)
                if any(p.grad is not None for p in world.forecast.parameters()):
                    raise RuntimeError('Frozen DSENet unexpectedly received a gradient')
                optimizer.step();completed_steps=step
                record=dict(step=step,loss=float(loss),gradient_norm=float(norm),event_groups=event_counts,
                            sample_indices=sample_indices,
                            components={kind:{key:float(value) for key,value in part.items()} for kind,part in losses.items()},
                            seconds=time.time()-begin)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if step==1 or step%config['print_every']==0:
                    print(json.dumps(dict(event='train',**record),allow_nan=False),flush=True)
                if step%config['validation_every']==0 or step==config['steps']:
                    world.eval()
                    with torch.no_grad():
                        validation={kind:validate_dataset(world,datasets[kind+'_validation'],kind=='paired',out,step,config,args.device)
                                    for kind in ('natural','paired')}
                    score=composite(validation['natural']['loss_components'],validation['paired']['loss_components'],config)
                    if not math.isfinite(score):raise FloatingPointError('Nonfinite validation score')
                    if frozen_state_digest(world.forecast)!=frozen_hash:
                        raise RuntimeError('Frozen forecast parameters or buffers changed during training')
                    validation.update(step=step,selection_score=score,split='world_validation',
                                      calibration='diagnostics only; no fitted post-hoc recalibration',seconds=time.time()-begin)
                    write_json(out/'validation'/('step%07d.json'%step),validation)
                    checkpoint=out/'checkpoints'/('step%07d.pt'%step)
                    save_checkpoint(checkpoint,world,optimizer,rng,step,config,validation,provenance_hash)
                    if score<best:
                        best=score;best_step=step;shutil.copyfile(checkpoint,out/'best.pt')
                    write_json(out/'selection.json',dict(best_step=best_step,best_score=best,
                                                        selected_on='world_validation',metric=config['selection'],
                                                        candidate_only=True,confirmation_not_used=True))
                    print(json.dumps(dict(event='world_validation',step=step,score=score,best_step=best_step)),flush=True)
        shutil.copyfile(out/'checkpoints'/('step%07d.pt'%config['steps']),out/'last.pt')
        write_json(out/'completion.json',dict(status='training_completed_candidate_only',steps=completed_steps,
                                               configured_steps=config['configured_steps'],budget_override=config['budget_override'],
                                               best_step=best_step,best_score=best,selected_on='world_validation',
                                               frozen_forecast_state_sha256=frozen_hash,
                                               closed_loop_validation='not_run_by_this_trainer',seconds=time.time()-begin,
                                               best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt')))
    except Exception as error:
        failed_checkpoint=None;checkpoint_error=None
        if 'world' in locals() and 'optimizer' in locals() and 'provenance_hash' in locals():
            try:
                save_checkpoint(out/'checkpoints'/'failed_state.pt',world,optimizer,rng,completed_steps,
                                config,None,provenance_hash)
                failed_checkpoint='checkpoints/failed_state.pt'
            except Exception as save_error:
                checkpoint_error=repr(save_error)
        write_json(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                                            failed_checkpoint=failed_checkpoint,checkpoint_save_error=checkpoint_error,
                                            completed_steps=completed_steps,seconds=time.time()-begin))
        raise


if __name__=='__main__':
    main()
