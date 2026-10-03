"""Equal-size Loop and frozen simulation batches for the unchanged R03 model."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / 'RL_DITR创新_2026-09-16'
sys.path.insert(0, str(REFERENCE))
from ditr_data import Data

FIELDS = ('state', 'action', 'target', 'mask', 'next_state', 'final_state',
          'k', 'length', 'terminal')


def numpy_batch(raw):
    result = {}
    for key in FIELDS:
        value = raw.get(key)
        if key == 'terminal' and value is None:
            value = np.zeros(len(raw['state']), dtype=bool)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        result[key] = np.asarray(value)
    return result


class MixedData:
    """Use every Loop origin as eligible; no filtering on sparse observations."""
    def __init__(self, shared_root, seed=260915, horizon=12, verify=True):
        from shared_data import SharedReplay
        self.loop = Data('train')
        self.shared = SharedReplay(root=Path(shared_root), device='cpu', verify=verify)
        self.seed = seed
        self.horizon = horizon
        self.loop_patients_seen = set()
        self.loop_epochs_started = 0
        self.sim_visits = 0
        self.rng = np.random.default_rng(seed)
        self.loop_stream = None
        self.pending = None
        self.sim_order = np.empty(0, dtype=np.int64)
        self.sim_cursor = 0

    def _loop_batch(self, count):
        chunks = []
        needed = count
        while needed:
            if self.pending is None:
                if self.loop_stream is None:
                    self.loop_stream = iter(self.loop.batches(
                        count, self.seed + self.loop_epochs_started, self.horizon))
                    self.loop_epochs_started += 1
                try:
                    raw = next(self.loop_stream)
                except StopIteration:
                    self.loop_stream = None
                    continue
                self.loop_patients_seen.add(raw['patient'])
                self.pending = numpy_batch(raw)
            take = min(needed, len(self.pending['state']))
            chunks.append({k: v[:take] for k, v in self.pending.items()})
            self.pending = ({k: v[take:] for k, v in self.pending.items()}
                            if take < len(self.pending['state']) else None)
            needed -= take
        return {k: np.concatenate([c[k] for c in chunks]) for k in FIELDS}

    def _sim_indices(self, count):
        chunks = []
        while count:
            if self.sim_cursor == len(self.sim_order):
                self.sim_order = self.rng.permutation(self.shared.size)
                self.sim_cursor = 0
            take = min(count, len(self.sim_order) - self.sim_cursor)
            chunks.append(self.sim_order[self.sim_cursor:self.sim_cursor + take])
            self.sim_cursor += take
            count -= take
        return np.concatenate(chunks)

    def batch(self, batch_size=128):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('A mixed batch must have two equally sized halves')
        half = batch_size // 2
        loop = self._loop_batch(half)
        sim = numpy_batch(self.shared.ditr_batch(
            self._sim_indices(half), horizon=self.horizon, rng=self.rng))
        self.sim_visits += half
        result = {k: np.concatenate([loop[k], sim[k]]) for k in FIELDS}
        expected = np.arange(self.horizon)[None, :] < result['length'][:, None]
        np.testing.assert_array_equal(result['mask'], expected)
        assert result['state'].shape == (batch_size, 72, 22)
        assert result['action'].shape == (batch_size, self.horizon)
        assert np.all(result['length'] > 0)
        assert result['terminal'].dtype == bool
        for key in ('state', 'action', 'target', 'next_state', 'final_state'):
            if not np.isfinite(result[key]).all():
                raise ValueError('Nonfinite mixed field: ' + key)
        result['source_is_simulation'] = np.arange(batch_size) >= half
        return result
