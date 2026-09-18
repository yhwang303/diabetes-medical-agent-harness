"""Patient-isolated masks, unchanged 72x22 histories and train-only scaling."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'RL训练_2026-09-15'))
from data import Data as OriginalData, tensor

class Data(OriginalData):
    def __init__(self, split):
        if split not in ('train', 'validation'):
            raise ValueError('Development training loader cannot open sealed data')
        super().__init__(split)
        for pid, d in self.patients.items():
            with np.load(ROOT / 'forecast_masks' / split / (pid + '.npz')) as f:
                np.testing.assert_array_equal(f['starts'], d['starts'])
                d['episode'] = f['episode']
                d['observed'] = f['observed']

    def batch(self, pid, positions, horizon=48, rng=None):
        d = self.patients[pid]
        starts = d['starts'][positions].astype('int64')
        row = starts[:, None] + np.arange(1, horizon + 1)[None]
        clipped = np.minimum(row, len(d['features']) - 1)
        mask = (row < len(d['features'])) & (d['episode'][clipped] == d['episode'][starts, None]) & d['observed'][clipped]
        target = np.where(mask, d['glucose'][clipped], 0).astype('float32')
        assert np.isfinite(target).all()
        return {'state': self.history(d, starts), 'target': target, 'mask': mask,
                'patient': pid, 'row': starts}
