"""R03 recursive categorical continuation on equal Loop/S minibatches.

Fixed final checkpoints; no simulator outcome participates in selection.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
REFERENCE = PROJECT / 'RL_DITR创新_2026-09-16'
sys.path.insert(0, str(REFERENCE))
from ditr_model import DITRAgent, status_score
from ditr_losses import patient_loss, discounted_return, GAMMA
from ditr_data import tensor
from ditr_shared_data import MixedData


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def historical_target(patient, batch):
    """Only genuine environment termination zeroes bootstrap; truncation does not."""
    with torch.no_grad():
        end = patient.value(patient.encode(batch['final_state'])[:, -1]).squeeze(-1)
        end = end.masked_fill(batch['terminal'], 0)
        return discounted_return(status_score(batch['target']) / 12,
                                 batch['mask'], end)


def corrected_policy_loss(agent, batch, horizon=12):
    """R03 legacy objective, with the same true-terminal fix as patient_loss."""
    with torch.no_grad():
        z = agent.patient.encode(batch['state'])[:, -1]
        historical_return = historical_target(agent.patient, batch)
    logp = agent.policy.log_prob(z, batch['action'][:, 0])
    historical = -(historical_return * logp).mean()
    logps, returns, _ = agent.imagine(batch['state'], horizon)
    imagined = -(returns * logps).mean()
    mean, std = agent.policy.parameters_at(z)
    supervised = (mean - batch['action'][:, 0]).square().mean()
    return historical + imagined + supervised, {
        'historical_policy_loss': historical,
        'imagined_policy_loss': imagined,
        'supervised_action_mse': supervised,
        'mean_std': std.mean(),
        'imagined_return': returns.mean(),
        'historical_return_min': historical_return.min(),
        'historical_negative_fraction': (historical_return < 0).float().mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--smoke-name')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    if args.smoke_steps:
        if not args.smoke_name or not 1 <= args.smoke_steps <= 10:
            raise ValueError('Smoke requires a distinct name and 1-10 steps per stage')
        cfg['smoke'] = True
        cfg['name'] = args.smoke_name
        cfg['patient']['steps'] = cfg['policy']['steps'] = args.smoke_steps
        cfg['log_every'] = 1
    if cfg['seed'] != 260915 or cfg['horizon'] != 12:
        raise ValueError('Frozen R03 seed/horizon contract differs')
    if cfg['model'].get('transition_mode', 'recursive') != 'recursive':
        raise ValueError('This continuation must retain recursive R03 transitions')
    if not cfg['model']['categorical_heads']:
        raise ValueError('Categorical R03 reward/value supports must remain')
    shared_root = PROJECT / cfg['shared_root']
    manifest_path = shared_root / 'manifest.json'
    if not cfg.get('smoke', False):
        expected = cfg.get('shared_manifest_sha256')
        if not expected or sha(manifest_path) != expected:
            raise ValueError('Formal training requires the explicitly frozen S manifest hash')
    out = ROOT / 'results' / cfg['name']
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', cfg)
    sources = out / 'sources'
    sources.mkdir()
    source_paths = [Path(__file__).resolve(), ROOT / 'ditr_shared_data.py', ROOT / 'shared_data.py',
                    REFERENCE / 'ditr_model.py', REFERENCE / 'ditr_losses.py',
                    REFERENCE / 'ditr_data.py', PROJECT / 'RL训练_2026-09-15/data.py']
    for path in source_paths:
        name = path.name if path.name != 'data.py' else 'frozen_loop_data.py'
        shutil.copy2(path, sources / name)
    checkpoint_path = PROJECT / cfg['initialize_checkpoint']
    checkpoint_hash = sha(checkpoint_path)
    if cfg.get('initialize_sha256') and checkpoint_hash != cfg['initialize_sha256']:
        raise ValueError('R03 initializer differs from frozen hash')
    torch.set_num_threads(cfg.get('threads', 4))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    agent = DITRAgent(**cfg['model']).cuda()
    ck = torch.load(checkpoint_path, map_location='cpu')
    if ck['stage'] != 'policy' or not ck['config']['model']['categorical_heads']:
        raise ValueError('Initializer is not the R03 policy-stage categorical checkpoint')
    if ck['config']['model'].get('transition_mode', 'recursive') != 'recursive':
        raise ValueError('Initializer is not recursive R03')
    agent.load_state_dict(ck['agent'], strict=True)
    target = copy.deepcopy(agent.patient).eval().requires_grad_(False)
    target.load_state_dict(ck['target_patient'], strict=True)
    data = MixedData(shared_root, cfg['seed'], cfg['horizon'])
    if data.loop.total != 1653421 or len(data.loop.patients) != 225:
        raise ValueError('Frozen complete Loop training population differs')
    source_hashes = {str(p.relative_to(PROJECT)): sha(p) for p in source_paths}
    manifest_hash = sha(manifest_path)
    write(out / 'provenance.json', {
        'source_sha256': source_hashes,
        'input_config_sha256': sha(config_path), 'effective_config_sha256': sha(out / 'config.json'),
        'initialize_sha256': checkpoint_hash,
        'initialize_stage': ck['stage'], 'initialize_step': ck['step'],
        'optimizer_restarted': True, 'ema_restored_from_original_checkpoint': True,
        'shared_manifest_sha256': manifest_hash,
        'shared_manifest': json.loads(manifest_path.read_text()),
        'loop_manifest': data.loop.manifest,
        'loop_files_sha256': {p.name: sha(p) for p in sorted(data.loop.root.glob('*.npz'))},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'gamma': GAMMA, 'label_contract': 'CGM mmol for original R03 glucose/status targets',
        'selection': 'fixed last patient then fixed last policy checkpoint',
        'clinical_ready': False,
    })
    started = time.time()
    stage, step = 'initialization', 0
    log = (out / 'history.jsonl').open('a', buffering=1)

    def event(item):
        item['elapsed_seconds'] = time.time() - started
        line = json.dumps(item, allow_nan=False)
        log.write(line + '\n')
        print(line, flush=True)

    def save(stage, step, optimizer):
        path = out / (stage + '_last.pt')
        temp = path.with_suffix('.tmp')
        torch.save({'agent': agent.state_dict(), 'target_patient': target.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': cfg,
                    'stage': stage, 'step': step,
                    'samples': step * cfg['batch_size'],
                    'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all(),
                    'rng_numpy': np.random.get_state(),
                    'continuation': 'new budget; exact mid-run resume not implemented'}, temp)
        temp.replace(path)
        write(out / (stage + '_checkpoint.json'), {'path': path.name, 'sha256': sha(path)})

    try:
        for stage in ('patient', 'policy'):
            agent.patient.requires_grad_(stage == 'patient')
            agent.policy.requires_grad_(stage == 'policy')
            parameters = list(agent.patient.parameters() if stage == 'patient'
                              else agent.policy.parameters())
            optimizer = torch.optim.Adam(parameters, lr=cfg[stage]['lr'],
                                         weight_decay=cfg['weight_decay'])
            window = []
            stage_start = time.time()
            event({'event': 'stage_start', 'stage': stage})
            for step in range(1, cfg[stage]['steps'] + 1):
                batch = tensor(data.batch(cfg['batch_size']))
                agent.train()
                agent.patient.train(stage == 'patient')
                optimizer.zero_grad(set_to_none=True)
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                    enable_mem_efficient=False):
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                        loss, parts = (patient_loss(agent.patient, target, batch, cfg['mu'])
                                       if stage == 'patient' else
                                       corrected_policy_loss(agent, batch, cfg['horizon']))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite ' + stage + ' loss')
                    loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'])
                if not torch.isfinite(grad):
                    raise RuntimeError('Nonfinite ' + stage + ' gradient')
                optimizer.step()
                if stage == 'patient':
                    with torch.no_grad():
                        for dst, src in zip(target.parameters(), agent.patient.parameters()):
                            dst.lerp_(src, cfg['target_tau'])
                window.append({'loss': float(loss), 'grad_norm': float(grad),
                               'terminal_count': float(batch['terminal'].sum()),
                               **{key: float(value) for key, value in parts.items()}})
                if step % cfg['log_every'] == 0 or step == cfg[stage]['steps']:
                    event({'event': 'train', 'stage': stage, 'step': step,
                           'samples_each_domain': step * cfg['batch_size'] // 2,
                           'loop_patients_seen': len(data.loop_patients_seen),
                           'seconds_per_step': (time.time() - stage_start) / step,
                           'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9,
                           **{key: float(np.mean([r[key] for r in window]))
                              for key in window[0]}})
                    window = []
            save(stage, step, optimizer)
            event({'event': 'stage_complete', 'stage': stage, 'step': step})
        if source_hashes != {str(p.relative_to(PROJECT)): sha(p) for p in source_paths}:
            raise RuntimeError('Training source changed during this run')
        if sha(manifest_path) != manifest_hash or sha(checkpoint_path) != checkpoint_hash:
            raise RuntimeError('Frozen manifest or original initializer changed during this run')
        write(out / 'completion.json', {
            'status': 'fixed_training_budget_finished_control_evaluation_pending',
            'stages': {s: cfg[s]['steps'] for s in ('patient', 'policy')},
            'wall_seconds': time.time() - started,
            'simulation_sample_visits': data.sim_visits,
            'loop_patients_seen': len(data.loop_patients_seen),
            'source_manifest_and_initializer_unchanged': True,
            'smoke_only': cfg.get('smoke', False), 'clinical_ready': False,
        })
    except Exception as error:
        write(out / 'failure.json', {'stage': stage, 'step': step,
              'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
