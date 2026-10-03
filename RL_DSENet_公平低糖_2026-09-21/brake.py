"""Observable-only research controls; not the clinical SSM implementation."""
import numpy as np

MEAN=7.602434716830251
SCALE=2.9487731123159437

def signals(history):
    x=np.asarray(history,dtype=float)
    if x.shape!=(72,22) or not np.isfinite(x).all() or not (x[-6:,5]>.5).all():
        raise ValueError('Six current CGM samples required')
    g=(x[-6:,0]*SCALE+MEAN)*18
    t=np.arange(6)*5.;slope=float(np.dot(t-t.mean(),g-g.mean())/np.sum((t-t.mean())**2))
    current=float(g[-1]);filtered=float(g[-2:].mean())
    return current,filtered,slope

def attenuation(glucose):
    if glucose>=112.5:return 1.,0.
    if glucose<=20:return 1./101,100.
    risk=10*(1.509*(np.log(glucose)**1.084-5.381))**2
    return 1./(1.+risk),float(risk)

class Brake:
    def __init__(self,config):self.config=dict(config);self.states={}
    def apply(self,key,history,raw):
        raw=float(raw);assert np.isfinite(raw) and 0<=raw<=20
        g,filtered,slope=signals(history);c=self.config
        estimate=filtered+c['slope_minutes']*slope
        phi,risk=attenuation(estimate)
        # Patek et al. power brakes are removed when the trend becomes nonnegative.
        active=slope<0 and phi<1
        applied=raw*phi if active else raw
        return float(applied),dict(current_cgm=g,filtered_cgm=filtered,slope=slope,estimate=estimate,
                                   attenuation=phi if active else 1.,risk=risk,active=bool(active),raw_u_h=raw)
