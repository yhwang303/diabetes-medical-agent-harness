"""Reuse the frozen complete Loop dataset; add observed bootstrap state only."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_loop_data', ROOT.parent / 'RL训练_2026-09-15/data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tensor = module.tensor

class Data(module.Data):
 def __init__(self,split,long_horizon=False):
  super().__init__(split)
  if long_horizon:
   for pid,d in self.patients.items():
    length=np.load(ROOT/'long_horizons'/split/(pid+'.npy'))
    np.testing.assert_array_equal(np.minimum(length,12),d['length']);d['length']=length
 def batch(self,pid,positions,horizon=12,rng=None):
  b=super().batch(pid,positions,horizon,rng)
  b['final_state']=self.history(self.patients[pid],b['row']+b['length'])
  return b
