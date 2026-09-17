from audit_trials import *
res={}
for ds in ('dclp3','iobp2'):
 base=ROOT/'datasets/extracted'/ds/('Data Files' if ds=='dclp3' else 'Data Tables')
 age=read(base/('DiabScreening_a.txt' if ds=='dclp3' else 'IOBP2PtRoster.txt'));adults=set(age.loc[age['AgeAtEnrollment' if ds=='dclp3' else 'AgeAsofEnrollDt']>=18,'PtID']);r={}
 for p in sorted(base.glob('*.txt')):
  head=read(p,nrows=0);cols=[c for c in head if any(x in c.lower() for x in ['carb','meal','weight'])]
  if not cols or p.stat().st_size>5000000:continue
  d=read(p);pid='PtID' if 'PtID' in d else 'DeidentID'; rr={}
  for c in cols:
   v=d[c].notna();num=pd.to_numeric(d[c],errors='coerce');rr[c]={'nonmissing_rows':int(v.sum()),'patients_nonmissing':d.loc[v,pid].nunique(),'adult_patients_nonmissing':d.loc[v&d[pid].isin(adults),pid].nunique(),'numeric_positive_rows':int((num>0).sum()),'adult_patients_positive':d.loc[(num>0)&d[pid].isin(adults),pid].nunique(),'values_if_small_cardinality':counts(d[c]) if d[c].nunique()<20 else None,'numeric_quantiles':quant(num)}
  r[p.name]=rr
 ins=read(base/('Insulin_a.txt' if ds=='dclp3' else 'IOBP2Insulin.txt'));name='ParentInsulinListID' if ds=='dclp3' else 'InsulinName';r['insulin_summary']={'all_name_rows':counts(ins[name]),'adult_name_rows':counts(ins.loc[ins.PtID.isin(adults),name]),'routes':counts(ins.InsRoute),'adult_pump_names_unique_patients':ins.loc[ins.PtID.isin(adults)&(ins.InsRoute=='Pump')].groupby(name).PtID.nunique().to_dict(),'start_unknown_rows':int(ins.InsTypeStartDt.isna().sum()),'stop_unknown_rows':int(ins.InsTypeStopDt.isna().sum())}
 if ds=='iobp2':
  for name in ['IOBP2ManualInsulinInj.txt','IOBP2DiscontiLet.txt','IOBP2BasalRtChg.txt','IOBP2DeviceDtTmVer.txt','IOBP2ContDeviceDtTmVer.txt']:
   d=read(base/name);r[name]={'rows':len(d),'patients':d.PtID.nunique(),'adult_patients':d.loc[d.PtID.isin(adults),'PtID'].nunique(),'nonmissing_counts':d.notna().sum().to_dict()}
 res[ds]=r
 dump('covariates_audit.json',res)
