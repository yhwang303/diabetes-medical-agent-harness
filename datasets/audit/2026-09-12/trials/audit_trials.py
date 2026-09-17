"""Read-only raw-event audit; no interpolation, resampling or training dataset output."""
from pathlib import Path
import pandas as pd, numpy as np, json, collections, sys
ROOT=Path.cwd(); OUT=ROOT/'datasets/audit/2026-09-12/trials'
def enc(p):
 with p.open('rb') as f:return 'utf-16' if f.read(2)==b'\xff\xfe' else 'utf-8-sig'
def read(p,**kw):return pd.read_csv(p,sep='|',encoding=enc(p),low_memory=False,**kw)
def clean(o):
 if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [clean(v) for v in o]
 if isinstance(o,(np.integer,)):return int(o)
 if isinstance(o,(np.floating,)):return float(o) if np.isfinite(o) else None
 if isinstance(o,np.ndarray):return clean(o.tolist())
 if isinstance(o,set):return clean(sorted(o))
 return o
def dump(name,o):(OUT/name).write_text(json.dumps(clean(o),ensure_ascii=False,indent=2,allow_nan=False))
def parse_iobp_time(s):
 t=pd.to_datetime(s,format='%m/%d/%Y %I:%M:%S %p',errors='coerce')
 v=t.isna()
 if v.any(): t.loc[v]=pd.to_datetime(s.loc[v],format='%m/%d/%Y',errors='coerce')
 return t
def counts(s):return {str(k):int(v) for k,v in s.fillna('<MISSING>').value_counts(dropna=False).items()}
def quant(s):
 s=pd.to_numeric(pd.Series(s),errors='coerce').dropna()
 return {str(k):float(v) for k,v in s.quantile([0,.01,.25,.5,.75,.95,.99,1]).items()} if len(s) else {}
def small_tables(ds):
 base=ROOT/'datasets/extracted'/ds/('Data Files' if ds=='dclp3' else 'Data Tables');res={};frames={}
 for p in sorted(base.glob('*.txt')):
  if p.stat().st_size>5000000:continue
  d=read(p); pid='PtID' if 'PtID' in d else 'DeidentID' if 'DeidentID' in d else None
  info={'rows':len(d),'patients':d[pid].nunique() if pid else None,'columns':list(d.columns),'missing':d.isna().sum().to_dict()}
  if not any(x in p.name.lower() for x in ['ps','fear','inspire','avoid','confidence','usability','technology','personality','distress','clarke']):
   info['low_cardinality_values']={c:counts(d[c]) for c in d if d[c].nunique()<=45 and c not in ['PtID','RecID','ParentLoginVisitID','DeidentID']}
  res[p.name]=info;frames[p.name]=d
 dump(f'{ds}_small_tables.json',res);return frames

def time_stats(d,pid,tcol,validcol=None):
 """Audit within exact original timestamps only; duplicate removal is diagnostic, not a cleaned output."""
 fmt='%d%b%y:%H:%M:%S' if d[tcol].dtype==object and str(d[tcol].dropna().iloc[0])[2:5].isalpha() else 'mixed'
 ts=pd.to_datetime(d[tcol],format=fmt,errors='coerce'); v=ts.notna()
 q=pd.DataFrame({'pid':d[pid].values,'t':ts.values});q=q.loc[v].sort_values(['pid','t']);dup=int(q.duplicated(['pid','t']).sum());q=q.drop_duplicates(['pid','t'])
 dt=q.groupby('pid').t.diff().dt.total_seconds();sp=q.groupby('pid').t.agg(['min','max','size']);days=(sp['max']-sp['min']).dt.total_seconds()/86400
 out={'invalid_time_rows':int((~v).sum()),'duplicate_patient_time_rows':dup,'unique_patient_timestamps':len(q),'patient_span_days_quantiles':quant(days),'sum_patient_span_days':float(days.sum()),'gap_seconds_quantiles':quant(dt),'gaps_gt_7_5min':int((dt>450).sum()),'gaps_gt_30min':int((dt>1800).sum()),'gaps_gt_6h':int((dt>21600).sum()),'five_minute_differences_exact':int((dt==300).sum()),'per_patient':{str(k):{'n':int(r['size']),'span_days':float(days.loc[k])} for k,r in sp.iterrows()}}
 return out

def run_dclp():
 ds='dclp3';fs=small_tables(ds);base=ROOT/'datasets/extracted'/ds/'Data Files'
 roster=fs['PtRoster_a.txt'];screen=fs['DiabScreening_a.txt']; adults=set(screen.loc[screen.AgeAtEnrollment>=18,'PtID']);allage=screen[['PtID','AgeAtEnrollment']].drop_duplicates()
 result={'cohort':{'roster_patients':roster.PtID.nunique(),'age_known_patients':allage.PtID.nunique(),'adults':len(adults),'age_quantiles':quant(screen.AgeAtEnrollment),'treatment_groups':counts(roster.trtGroup),'adult_treatment_groups':counts(roster.loc[roster.PtID.isin(adults),'trtGroup'])},'tables':{}}
 ids={}
 for name,val,tcol in [('Pump_BasalRateChange.txt','CommandedBasalRate','DataDtTm'),('Pump_BolusDelivered.txt','BolusAmount','DataDtTm'),('Pump_CGMGlucoseValue.txt','CGMValue','DataDtTm'),('cgm.txt','CGM','DataDtTm'),('DexcomClarityCGM_a.txt','CGM','DataDtTm'),('OtherCGM_a.txt','CGM','DataDtTm')]:
  print('reading',name,flush=True);d=read(base/name);ids[name]=set(d.PtID.unique())
  adj=next((x for x in ['DataDtTm_adjusted','DataDtTm_adj'] if x in d),None)
  info={'rows':len(d),'patients':d.PtID.nunique(),'adult_patients':len(ids[name]&adults),'adult_rows':int(d.PtID.isin(adults).sum()),'columns':list(d.columns),'missing':d.isna().sum().to_dict(),'value_quantiles':quant(d[val]),'negative_values':int((d[val]<0).sum()),'zero_values':int((d[val]==0).sum()),'exact_duplicates_excluding_RecID':int(d.drop(columns=['RecID'],errors='ignore').duplicated().sum()),'raw_time':time_stats(d,'PtID',tcol)}
  if adj:
   info['adjusted_time_nonmissing_rows']=int(d[adj].notna().sum()); dd=d[['PtID',tcol,adj]].copy();dd['audit_time']=dd[adj].fillna(dd[tcol]);info['source_adjusted_else_raw_time']=time_stats(dd,'PtID','audit_time')
  for col in ['BolusType','HighLowIndicator','Period','EventSubType']:
   if col in d:info[col]=counts(d[col])
  # Same timestamp conflicting numerical values, regardless RecID.
  info['patient_time_multiple_distinct_values']=int((d.groupby(['PtID',tcol])[val].nunique()>1).sum())
  if name=='Pump_BasalRateChange.txt':
   info['rate_gt25_rows']=int((d[val]>25).sum());info['missing_duration_end_suspend_columns']=not any('duration' in c.lower() or 'end' in c.lower() or 'suspend' in c.lower() for c in d)
  result['tables'][name]=info;dump('dclp3_audit.json',result);del d
 both=ids['Pump_BasalRateChange.txt']&ids['Pump_BolusDelivered.txt']&ids['cgm.txt'];result['cohort']['cgm_basal_bolus_intersection']=len(both);result['cohort']['adult_cgm_basal_bolus_intersection']=len(both&adults);result['cohort']['intersection_adult_treatment_groups']=counts(roster.loc[roster.PtID.isin(both&adults),'trtGroup']);result['cohort']['adult_cgm_basal_bolus_ids']=both&adults
 dump('dclp3_audit.json',result)


def run_iobp():
 ds='iobp2';fs=small_tables(ds);base=ROOT/'datasets/extracted'/ds/'Data Tables';roster=fs['IOBP2PtRoster.txt'];adults=set(roster.loc[roster.AgeAsofEnrollDt>=18,'PtID'])
 result={'cohort':{'roster_patients':roster.PtID.nunique(),'adult_patients':len(adults),'age_quantiles':quant(roster.AgeAsofEnrollDt),'treatment_groups':counts(roster.TrtGroup),'adult_treatment_groups':counts(roster.loc[roster.PtID.isin(adults),'TrtGroup'])},'iLet':{}}
 # Aggregate each chunk, keep only numeric/time columns needed for chronological diagnostics.
 parts=[];missing=collections.Counter(); cats=collections.defaultdict(collections.Counter);rows=0; comp=collections.Counter(); mins={};maxs={};pats=set();adrows=0
 keep=['PtID','UploadIndex','DeviceDtTm','CGMVal','InsComp','InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolus','MealBolusDelivPrev','InsDelivAvail']
 for d in read(base/'IOBP2DeviceiLet.txt',chunksize=200000):
  rows+=len(d);adrows+=int(d.PtID.isin(adults).sum());pats.update(d.PtID.unique());missing.update(d.isna().sum().to_dict())
  for c in ['InsDelivAvail','InfSiteChg','GlucDelivAvail','GlucBurst','BasalScale','BGTarget','MealSize','MealTimeOfDay']:cats[c].update(counts(d[c]))
  for c in ['CGMVal','InsComp','InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolus','MealBolusDelivPrev']:
   mins[c]=min(mins.get(c,np.inf),d[c].min());maxs[c]=max(maxs.get(c,-np.inf),d[c].max());comp[c+'_negative']+=int((d[c]<0).sum());comp[c+'_zero']+=int((d[c]==0).sum())
  cols=['InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolusDelivPrev'];v=d[cols].notna().all(axis=1);comp['complete_delivered_components']+=int(v.sum())
  err2=(d.InsDelivPrev-d.BasalDelivPrev-d.BolusDelivPrev).abs();err3=(d.InsDelivPrev-d.BasalDelivPrev-d.BolusDelivPrev-d.MealBolusDelivPrev).abs()
  comp['total_equals_basal_plus_bolus_tol_0_002']+=int((v&(err2<=.00200001)).sum());comp['total_equals_all_three_tol_0_002']+=int((v&(err3<=.00200001)).sum());comp['meal_positive_rows']+=int((d.MealBolusDelivPrev>0).sum());comp['meal_positive_total_eq_two']+=int(((d.MealBolusDelivPrev>0)&(err2<=.00200001)).sum());comp['meal_positive_total_eq_three']+=int(((d.MealBolusDelivPrev>0)&(err3<=.00200001)).sum())
  z=d[keep].copy();z['DeviceDtTm']=parse_iobp_time(z.DeviceDtTm);parts.append(z)
  if rows%1000000==0:print('ilet',rows,flush=True)
 d=pd.concat(parts,ignore_index=True);del parts
 info={'rows':rows,'patients':len(pats),'adult_patients':len(pats&adults),'adult_rows':adrows,'missing':dict(missing),'categories':{k:dict(v) for k,v in cats.items()},'min':mins,'max':maxs,'component_consistency':dict(comp),'original_timestamp_stats':time_stats(d,'PtID','DeviceDtTm')}
 key=['PtID','DeviceDtTm','UploadIndex'];core=['CGMVal','InsComp','InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolus','MealBolusDelivPrev','InsDelivAvail']
 info['duplicate_patient_time_step_rows']=int(d.duplicated(key).sum());info['duplicate_all_selected_fields_rows']=int(d.duplicated(key+core).sum());dups=d[d.duplicated(key,keep=False)];info['duplicate_key_groups_with_any_core_conflict']=int((dups.groupby(key)[core].nunique(dropna=False).max(axis=1)>1).sum()) if len(dups) else 0
 # Remove exact selected-event duplicates only in-memory to assess clock/step continuity, never output cleaned records.
 q=d.drop_duplicates(key+core).sort_values(['PtID','DeviceDtTm','UploadIndex']);dt=q.groupby('PtID').DeviceDtTm.diff().dt.total_seconds();di=q.groupby('PtID').UploadIndex.diff();
 info['diagnostic_unique_selected_rows']=len(q);info['index_diff_counts']={'eq1':int((di==1).sum()),'eq0':int((di==0).sum()),'negative':int((di<0).sum()),'gt1':int((di>1).sum())};info['consecutive_step_pairs']=int((di==1).sum());info['consecutive_step_time_300sec']=int(((di==1)&(dt==300)).sum());info['consecutive_step_time_within_1sec']=int(((di==1)&((dt-300).abs()<=1)).sum());info['consecutive_step_time_gt450sec']=int(((di==1)&(dt>450)).sum());info['consecutive_step_time_le0']=int(((di==1)&(dt<=0)).sum())
 same=(di==1)&((dt-300).abs()<=1);prevcomp=q.groupby('PtID').InsComp.shift();err=(q.InsDelivPrev-prevcomp).abs();info['computed_previous_vs_delivered']={'eligible_chronological_pairs':int(same.sum()),'abs_diff_le0_002':int((same&(err<=.00200001)).sum()),'abs_diff_gt0_05':int((same&(err>.05)).sum()),'abs_diff_quantiles':quant(err[same])}
 info['valid_cgm_40_400_rows']=int(q.CGMVal.between(40,400).sum());info['CGM_outside_40_400_rows']=int((~q.CGMVal.between(40,400)).sum());info['positive_delivery_when_channel_false']=int(((q.InsDelivAvail==False)&(q.InsDelivPrev>0)).sum())
 info['adult_candidate_consecutive_step_time_within1sec_cgm_both_valid_channel_true']=int((same&q.PtID.isin(adults)&q.CGMVal.between(40,400)&q.groupby('PtID').CGMVal.shift().between(40,400)&(q.InDelivAvail if 'InDelivAvail' in q else q.InsDelivAvail)).sum())
 result['iLet']=info;result['cohort']['adult_iLet_ids']=pats&adults;result['cohort']['adult_iLet_treatment_groups']=counts(roster.loc[roster.PtID.isin(pats&adults),'TrtGroup']);dump('iobp2_audit.json',result)
 del d,q,dups
 # CGM table includes multiple record types: summarize all, assess CGM numeric records separately.
 cparts=[];rtypes=collections.Counter();units=collections.Counter();miss=collections.Counter();n=0;cids=set()
 for c in read(base/'IOBP2DeviceCGM.txt',chunksize=250000):
  n+=len(c);rtypes.update(counts(c.RecordType));units.update(counts(c.Units));miss.update(c.isna().sum().to_dict());cids.update(c.PtID.unique());x=c.loc[c.RecordType=='CGM',['PtID','DeviceDtTm','InternalDtTm','Value']].copy();x.DeviceDtTm=parse_iobp_time(x.DeviceDtTm);cparts.append(x)
 c=pd.concat(cparts,ignore_index=True);del cparts
 result['cgm']={'all_rows':n,'all_patients':len(cids),'record_types':dict(rtypes),'units':dict(units),'missing':dict(miss),'cgm_rows':len(c),'cgm_patients':c.PtID.nunique(),'adult_cgm_patients':len(set(c.PtID)&adults),'value_quantiles':quant(c.Value),'time_stats':time_stats(c,'PtID','DeviceDtTm'),'same_patient_time_conflicting_glucose_groups':int((c.groupby(['PtID','DeviceDtTm']).Value.nunique()>1).sum())}
 result['cohort']['adult_cgm_ilet_intersection']=len(set(c.PtID)&pats&adults);dump('iobp2_audit.json',result)

if __name__=='__main__':
 if len(sys.argv)==1 or sys.argv[1]=='dclp3':run_dclp()
 if len(sys.argv)==1 or sys.argv[1]=='iobp2':run_iobp()
