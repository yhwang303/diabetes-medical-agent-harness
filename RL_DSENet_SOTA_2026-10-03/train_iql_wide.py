"""Fixed-budget, from-scratch IQL; no development evaluation or selection."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from iql_wide import IQL
from prepare_iql_replay import R, P, Replay, load_config, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int, help='Only 4 is allowed for a separate smoke run')
    parser.add_argument('--name', help='Required separate result directory name for smoke')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args(); config = load_config(args.config)
    smoke = args.steps is not None
    if smoke and (args.steps != 4 or not args.name or args.name == config['name']):
        raise ValueError('Smoke must be exactly --steps 4 with a separate --name')
    if not smoke and args.name is not None:
        raise ValueError('Formal training uses the frozen config name')
    name = args.name if smoke else config['name']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Result name must be one directory name')
    steps = 4 if smoke else config['updates']
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no implicit device fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    replay_path = (R/config['replay']).resolve()
    try:
        replay_path.relative_to(R/'data')
    except ValueError:
        raise ValueError('Replay must be inside the new research data directory')
    replay = Replay(replay_path, config)
    out = R/'results'/name; out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', config)
    provenance = dict(config_sha256=sha(args.config), replay_manifest_sha256=sha(replay_path/'manifest.json'),
                      replay_path=str(replay_path.relative_to(P)),
                      source_sha256=replay.manifest['source_sha256'],
                      raw_data_sha256=replay.manifest['raw_data_sha256'],
                      data_selection=replay.manifest['selection'],
                      episodes=replay.manifest['episodes'], transitions=replay.size,
                      training_seed=config['seed'], smoke=smoke, planned_updates=steps,
                      torch_version=torch.__version__, numpy_version=np.__version__,
                      runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                   compute_dtype='float32', matmul_tf32=False, cudnn_tf32=False,
                                   omp_num_threads=2, openblas_num_threads=2),
                      device=args.device, model_input='history + physiology + observed anchor; no IDs, hidden state or future',
                      checkpoint_selection='fixed final 20000, no development selection')
    write_json(out/'provenance.json', provenance)
    source_dir = out/'source'; source_dir.mkdir()
    for relative, digest in provenance['source_sha256'].items():
        source = P/relative; target = source_dir/relative
        if sha(source) != digest:
            raise ValueError('Source changed while freezing training provenance')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    agent = IQL(config).to(args.device); agent.optimizers(); agent.train()
    step = 0; started = time.time(); window = []; checkpoints = []
    try:
        with (out/'history.jsonl').open('x', buffering=1) as log:
            for step in range(1, steps+1):
                indices = rng.integers(0, replay.size, config['batch_size'])
                batch = {k: torch.as_tensor(v, device=args.device) for k, v in replay.batch(indices).items()}
                result = agent.update(batch); window.append(result)
                if step % 100 == 0 or step == steps:
                    record = {k: float(np.mean([item[k] for item in window])) for k in result}
                    record.update(step=step, sample_visits=step*config['batch_size'], seconds=time.time()-started)
                    log.write(json.dumps(record, allow_nan=False)+'\n'); print(json.dumps(record), flush=True)
                    window = []
                if (not smoke and step in config['checkpoints']) or (smoke and step == steps):
                    filename = 'smoke_step4.pt' if smoke else 'policy_%06d.pt' % step
                    path = out/filename
                    state = dict(schema=1, kind='IQL_wide', config=config, model=agent.state_dict(),
                                 optimizers=agent.optimizer_state(), step=step, smoke=smoke,
                                 final=(step == 20000 and not smoke), provenance=provenance,
                                 rng=dict(python=random.getstate(), numpy_legacy=np.random.get_state(),
                                          sampler=rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
                                          torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
                    with path.open('xb') as stream:
                        torch.save(state, stream)
                    checkpoints.append(dict(path=filename, step=step, sha256=sha(path)))
        write_json(out/'completion.json', dict(status='smoke_completed' if smoke else 'fixed_budget_completed',
                   steps=steps, seed=seed, sample_visits=steps*config['batch_size'],
                   seconds=time.time()-started, checkpoints=checkpoints,
                   final_checkpoint=None if smoke else 'policy_020000.pt',
                   development_or_confirmation_used=False))
    except Exception as error:
        write_json(out/'failure.json', dict(step=step, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
