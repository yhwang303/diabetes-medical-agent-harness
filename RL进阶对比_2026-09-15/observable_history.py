"""Common observable history adapter, independent of simulator and Torch runtimes."""
from pathlib import Path
import json
import numpy as np
NORMALIZER=Path(__file__).resolve().parent.parent/'Loop数据集/训练管线_v2/prepared/normalization.json'

class History:
 def __init__(self):
  norm=json.loads(NORMALIZER.read_text());names=['cgm_mmol_l','basal_u_prev_5min','bolus_recorded_u_prev_5min','carbs_recorded_g_prev_5min','exercise_event_count_prev_5min']
  self.mean=np.array([norm[n]['mean'] for n in names]);self.scale=np.array([norm[n]['scale'] for n in names]);self.rows=[];self.last=np.full(5,-np.inf)
 def append(self,minutes,cgm,basal_prev=None,bolus_prev=None,food_prev=None):
  values=np.array([cgm/18,basal_prev,bolus_prev,food_prev,None],dtype=float);observed=np.isfinite(values);self.last=np.where(observed,minutes,self.last);known=np.isfinite(self.last)
  age=np.where(known,np.log1p(np.minimum(np.maximum(minutes-self.last,0),1440)/60),0)
  x=np.concatenate([np.where(observed,(values-self.mean)/self.scale,0),observed,age,known]).astype('float32');self.rows.append(x)
 def state(self):
  if len(self.rows)<72:raise ValueError('Policy requested before real warmup history')
  clock=np.stack([np.ones(72)/12,np.arange(-71,1)/12],-1)
  return np.concatenate([np.array(self.rows[-72:]),clock],-1).astype('float32')

