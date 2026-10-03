import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent
P = R.parent.parent
B = P / 'RL_DSENet_2026-09-17'
OLD = P / 'RL_DITR创新_2026-09-16'
sys.path.insert(0, str(B))
from world_model import Patient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_world(path):
    ck = torch.load(path, map_location='cpu')
    c = ck['config']
    world = Patient(B/c['forecast_checkpoint'], P/c['context_checkpoint']).cuda().eval()
    world.load_state_dict(ck['model'])
    return world, ck


def history_clock(x):
    clock = np.stack([np.ones(72)/12, np.arange(-71, 1)/12], -1).astype('float32')
    return np.concatenate([x, np.broadcast_to(clock, (len(x), 72, 2))], -1).astype('float32')


def encode(world, histories, batch=128):
    zs, fs, rates = [], [], []
    with torch.no_grad():
        for i in range(0, len(histories), batch):
            x = torch.as_tensor(histories[i:i+batch], device='cuda')
            z, f = world.encode(x)
            zs.append(z.cpu().numpy()); fs.append(f.cpu().numpy())
            rates.append(world.reference_rate(x).cpu().numpy())
    return dict(z=np.concatenate(zs), forecast=np.concatenate(fs), rate=np.concatenate(rates))


def gpu(arrays):
    return {k: torch.as_tensor(v, device='cuda') for k, v in arrays.items()}
