"""Descriptive action-support diagnostics, not a causal identification proof."""
import numpy as np,json
from data import Data,ROOT
D=Data('train');rows=[];per=[]
for pid,d in D.patients.items():
 s=d['starts'].astype('int64');a=d['action'][s]
 # Most recent logged basal amount (normalized feature 1) converted to U/h.
 prev=(d['features'][s,1]*.14462788945609448+.09945811581924525)*12
 g=d['glucose'][s];g30=d['glucose'][np.maximum(s-6,0)];trend=np.where(np.isfinite(g30),g-g30,0)
 # Recorded history totals, missing=absence of RECORD, not verified absence of event.
 food=np.zeros(len(s));count=np.zeros(len(s))
 f=d['features'];recorded=f[:,8]>.5;values=np.where(recorded,f[:,3]*21.64806233191839+28.85985633360002,0);cumsum=np.r_[0,values.cumsum()];food=cumsum[s+1]-cumsum[s-71]
 # Explicit coarse bins, report sensitivity over two scales; not propensity estimation.
 rows.append(np.stack([g,trend,prev,food,a],1));per.append({'patient':pid,'n':len(a),'same_as_previous_recorded_rate_fraction':float(np.isclose(a,prev,atol=1e-5).mean()),'action_u_h_q05_q50_q95':np.quantile(a,[.05,.5,.95]).tolist(),'action_sd':float(a.std())})
X=np.concatenate(rows);result={'n':len(X),'patients':per,'coarse_groups':[],'causal_identification_proven':False,'description':'Within similar measured-history bins action variance is descriptive; unmeasured physiology, timing and deterministic Loop policy preclude an ignorability claim.'}
for scales in [[1.,.5,.5,20.],[.5,.25,.25,10.]]:
 key=np.floor(X[:,:4]/np.array(scales)).astype('int64');_,inv=np.unique(key,axis=0,return_inverse=True);n=np.bincount(inv);total=np.bincount(inv,weights=X[:,4]);sq=np.bincount(inv,weights=X[:,4]**2);sd=np.sqrt(np.maximum(sq/n-(total/n)**2,0));valid=n>=50
 result['coarse_groups'].append({'bin_widths_glucose_trend_prevrate_recordedcarbs':scales,'groups_n_atleast50':int(valid.sum()),'samples_in_these_groups':int(n[valid].sum()),'sample_weighted_mean_conditional_action_sd':float(np.average(sd[valid],weights=n[valid])),'fraction_of_supported_samples_in_sd_under_0_1_groups':float(n[valid&(sd<.1)].sum()/n[valid].sum())})
result['action_vs_next_glucose_change_correlation_is_confounded']=None
(ROOT/'results/action_support.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='patients'},indent=2))
