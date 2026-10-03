"""Common immutable public-simulation replay; Loop remains a separate source."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


class SharedReplay:
    def __init__(self, root=None, device='cuda', verify=True):
        self.root = Path(root) if root is not None else ROOT / 'shared_replay'
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        assert self.manifest['status'] == 'complete' and self.manifest['schema_version'] == 1
        if verify:
            for name, digest in self.manifest['array_sha256'].items():
                assert hashlib.sha256((self.root / name).read_bytes()).hexdigest() == digest, name
            assert hashlib.sha256((self.root / 'episodes.json').read_bytes()).hexdigest() == self.manifest['episodes_sha256']
        self.device = device
        self.arrays = {Path(name).stem: torch.from_numpy(np.load(self.root/name, allow_pickle=False)).to(device)
                       for name in self.manifest['array_sha256']}
        self.size = len(self.arrays['start'])
        assert self.size == self.manifest['origins']
        self.history_offsets = torch.arange(-71, 1, device=device)
        self.time = torch.stack([torch.ones(72, device=device)/12, self.history_offsets/12], -1).float()

    def history(self, rows):
        features = self.arrays['features'][rows[:, None] + self.history_offsets]
        return torch.cat([features, self.time.expand(len(rows), -1, -1)], -1)

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        a = self.arrays
        rows = a['start'][indices]
        fields = ['action', 'next_action', 'reward', 'q_valid', 'bootstrap_valid',
                  'next_action_available', 'terminal', 'time_limit', 'bg_reward',
                  'ppo_reward', 'bg_mg_dl', 'episode_remaining', 'episode_id']
        return dict(state=self.history(rows).flatten(1), next_state=self.history(rows+1).flatten(1),
                    **{k: a[k][indices].reshape(-1, 1) for k in fields})

    def ditr_batch(self, indices, horizon=12, rng=None):
        """Observed multi-step labels; only a reached true terminal zeros value.

        k uses the original zero-based target index convention. final_state is
        observed at the last included transition; horizon cuts are bootstrapped.
        """
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        a = self.arrays
        n = len(indices)
        length = a['episode_remaining'][indices].long().clamp_max(horizon)
        steps = torch.arange(horizon, device=self.device)
        mask = steps[None, :] < length[:, None]
        positions = indices[:, None] + torch.minimum(steps[None, :], length[:, None]-1)
        assert torch.equal(a['episode_id'][positions], a['episode_id'][indices, None].expand(-1, horizon))
        action = torch.where(mask, a['action_u_h'][positions], 0.)
        target = torch.where(mask, a['cgm_mmol_l'][positions], 0.)
        if rng is None:
            k = length
        else:
            sampled = rng.integers(0, 2**31, size=n)
            k = torch.as_tensor(sampled, device=self.device) % length + 1
        rows = a['start'][indices]
        terminal = a['terminal'][indices+length-1]
        return dict(state=self.history(rows), action=action, target=target, mask=mask,
                    next_state=self.history(rows+k), final_state=self.history(rows+length),
                    k=k-1, length=length, terminal=terminal, row=rows,
                    patient='shared_public_simulation', episode_id=a['episode_id'][indices],
                    bg_mg_dl=torch.where(mask, a['bg_mg_dl'][positions], 0.))
