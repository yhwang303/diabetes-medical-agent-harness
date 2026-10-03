"""Observed outcomes and missing-outcome bounds, never invented glucose tails."""
import numpy as np

def episodes(values,threshold=70,dt=5):
 """Intervals below threshold start an event after 15m; 15m recovery closes it."""
 events=[];start=None;low_run=0;recovery=0;active=False;low_minutes=0
 for i,value in enumerate(values):
  if value<threshold:
   if start is None:start=i
   low_run+=1;recovery=0;low_minutes+=dt
   if low_run*dt>=15:active=True
  else:
   low_run=0
   if start is not None:
    recovery+=1
    if not active:
     start=None;low_minutes=0;recovery=0
    elif recovery*dt>=15:
     events.append({'start_index':start,'end_index_exclusive':i+1-recovery,'low_minutes':low_minutes,'span_minutes':(i+1-recovery-start)*dt,'right_censored':False})
     start=None;low_minutes=0;recovery=0;active=False
 if active:events.append({'start_index':start,'end_index_exclusive':len(values),'low_minutes':low_minutes,'span_minutes':(len(values)-start)*dt,'right_censored':True})
 return events

def glucose_metrics(values,planned_intervals,dt=5):
 g=np.asarray(values,dtype=float);valid=np.isfinite(g)&(g>0);n=int(valid.sum());missing=planned_intervals-n
 if missing<0:raise ValueError('More observations than planned')
 if not n:return {'observed_intervals':0,'coverage_pct':0.,'tir_lower_bound_pct':0.,'tir_upper_bound_pct':100.,'invalid_intervals':int(len(g)-n)}
 # Nonfinite observations are explicitly counted; event paths are split at invalid intervals.
 observed=g[valid];u=1.509*(np.log(np.clip(observed,20,600))**1.084-5.381);risk=10*u*u
 low=np.where(u<=0,risk,0);high=np.where(u>=0,risk,0);low=np.where(observed<=20,100,np.where(observed>=600,0,low));high=np.where(observed>=600,100,np.where(observed<=20,0,high))
 tir=(observed>=70)&(observed<=180);mean=float(observed.mean());sd=float(observed.std());event_list=[];long54=[]
 # Segment boundaries prevent an invalid measurement from linking distinct events.
 starts=np.flatnonzero(valid&~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid&~np.r_[valid[1:],False])+1
 for s,e in zip(starts,ends):
  for event in episodes(g[s:e]):
   event['start_index']+=int(s);event['end_index_exclusive']+=int(s);event_list.append(event)
  lower=g[s:e]<54;ss=np.flatnonzero(lower&~np.r_[False,lower[:-1]]);ee=np.flatnonzero(lower&~np.r_[lower[1:],False])+1
  long54.extend([int((b-a)*dt) for a,b in zip(ss,ee) if (b-a)*dt>=120])
 return {'observed_intervals':n,'invalid_intervals':int(len(g)-n),'coverage_pct':100*n/planned_intervals,'tir_observed_pct':100*float(tir.mean()),'tir_lower_bound_pct':100*int(tir.sum())/planned_intervals,'tir_upper_bound_pct':100*(int(tir.sum())+missing)/planned_intervals,'tbr70_pct':100*float((observed<70).mean()),'tbr54_pct':100*float((observed<54).mean()),'tar180_pct':100*float((observed>180).mean()),'tar250_pct':100*float((observed>250).mean()),'mean_mg_dl':mean,'sd_mg_dl':sd,'cv_pct':100*sd/mean,'lbgi':float(low.mean()),'hbgi':float(high.mean()),'risk':float((low+high).mean()),'hypo_events':event_list,'hypo_events_per_observed_day':len(event_list)/(n*dt/1440),'max_hypo_low_minutes':max([e['low_minutes'] for e in event_list],default=0),'right_censored_hypo_events':sum(e['right_censored'] for e in event_list),'prolonged_under54_120min_events':len(long54)}

def summarize(records,planned_intervals,failed,dt=5):
 rows=[r for r in records if not r['warmup']];n=len(rows);days=n*dt/1440
 output={key:glucose_metrics([r[field] for r in rows],planned_intervals,dt) for key,field in [('bg','bg_mg_dl'),('cgm','cgm_mg_dl')]}
 rates=np.array([r['delivered_basal_u_h'] for r in rows]);basal=float(rates.sum()*dt/60);bolus=float(sum(r['bolus_u'] for r in rows))
 output.update({'failed':bool(failed),'planned_intervals':planned_intervals,'observed_intervals':n,'basal_u_observed':basal,'bolus_u_observed':bolus,'basal_u_per_observed_day':basal/days if days else None,'bolus_u_per_observed_day':bolus/days if days else None,'action_tv_per_observed_day':float(np.abs(np.diff(rates)).sum())/days if days else None,'pump_changed_fraction':float(np.mean([abs(r['requested_basal_u_h']-r['delivered_basal_u_h'])>1e-7 for r in rows])) if rows else None})
 return output
