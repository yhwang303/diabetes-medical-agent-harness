"""Deterministic evaluation of the fixed-final IQL wide-action baseline."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from iql_wide import IQL
from prepare_iql_replay import P, encode_observations, read_json, sha


class Worker:
    def __init__(self, checkpoint, device='cuda', allow_smoke=False):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            raise ValueError('--checkpoint must be absolute')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['kind'] != 'IQL_wide' or state['config']['seed'] != 260915:
            raise ValueError('Not the frozen IQL-wide training contract')
        if state['smoke']:
            if not allow_smoke or state['step'] != 4:
                raise ValueError('Smoke checkpoint requires explicit --allow-smoke')
        elif state['step'] != 20000 or not state['final']:
            raise ValueError('Only the fixed final 20000-update checkpoint is eligible')
        self.config = state['config']; self.provenance = state['provenance']
        config_path = checkpoint.parent/'config.json'
        provenance_path = checkpoint.parent/'provenance.json'
        if self.config != read_json(config_path) or self.provenance != read_json(provenance_path):
            raise ValueError('Checkpoint embedded configuration/provenance differs from its saved run')
        self.config_sha256 = sha(config_path)
        self.provenance_sha256 = sha(provenance_path)
        for relative, digest in self.provenance['source_sha256'].items():
            if sha(P/relative) != digest:
                raise ValueError('Frozen source/feature contract changed: '+relative)
        manifest = P/self.provenance['replay_path']/'manifest.json'
        if sha(manifest) != self.provenance['replay_manifest_sha256']:
            raise ValueError('Training data manifest changed')
        self.device = device
        self.agent = IQL(self.config).to(device).eval().requires_grad_(False)
        self.agent.load_state_dict(state['model'], strict=True)
        self.checkpoint_sha256 = sha(checkpoint); self.smoke = state['smoke']
        self.runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                            device=device, compute_dtype='float32', matmul_tf32=False, cudnn_tf32=False,
                            omp_num_threads=2, openblas_num_threads=2)

    def evaluate(self, history, anchors):
        anchor = np.asarray(anchors, dtype=np.float64)
        state = encode_observations(history, anchor, self.config)
        started = time.perf_counter()
        with torch.no_grad():
            normalized = self.agent.act(torch.as_tensor(state, device=self.device)).flatten().cpu().numpy()
        if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 1):
            raise ValueError('Nonfinite or unbounded IQL output')
        raw = anchor*(normalized.astype(np.float64)+1.)
        # Intrinsic declared global cap, not the old absolute-action projection.
        actions = np.minimum(raw, 20.)
        if (actions < 0).any() or (actions > np.minimum(20., 2*anchor)+1e-8).any():
            raise ValueError('IQL action outside common task support')
        return dict(actions=actions.tolist(), actions_u_h=actions.tolist(),
                    normalized_actions=normalized.tolist(), global_cap_applied=(raw > 20).tolist(),
                    batch_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    worker = Worker(args.checkpoint, args.device, args.allow_smoke)
    print(json.dumps(dict(ready=True, kind='IQL_wide', state_dim=1613, deterministic=True,
                          checkpoint_sha256=worker.checkpoint_sha256, config_sha256=worker.config_sha256,
                          provenance_sha256=worker.provenance_sha256, runtime=worker.runtime,
                          smoke=worker.smoke)), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('op') == 'close':
                print(json.dumps(dict(closed=True)), flush=True); break
            if request.get('op') != 'evaluate':
                raise ValueError('Only op=evaluate or op=close is accepted; no training state')
            print(json.dumps(worker.evaluate(request['history'], request['anchors']), allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps(dict(error=repr(error))), flush=True)


if __name__ == '__main__':
    main()
