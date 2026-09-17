from audit_trials import *
base=ROOT/'datasets/extracted/iobp2/Data Tables';roster=read(base/'IOBP2PtRoster.txt').set_index('PtID');ads=set(roster.index[roster.AgeAsofEnrollDt>=18]);g=roster.TrtGroup.to_dict();rand=pd.to_datetime(roster.RandDt,format='mixed');trans=pd.to_datetime(roster.TransRandDt,format='mixed');cnt=collections.Counter();meal=collections.Counter();errhist=collections.Counter();group=collections.defaultdict(collections.Counter);parts=[];unitstats={}
for d in read(base/'IOBP2DeviceiLet.txt',chunksize=200000):
 dateonly=~d.DeviceDtTm.str.contains(' ');cnt['date_only_timestamp_rows']+=int(dateonly.sum());t=parse_iobp_time(d.DeviceDtTm);cnt['unparseable_timestamps']+=int(t.isna().sum());grp=d.PtID.map(g);ageok=d.PtID.isin(ads);dt=(t-d.PtID.map(rand)).dt.total_seconds()/86400
 er=(d.InsDelivPrev-d.BasalDelivPrev-d.BolusDelivPrev-d.MealBolusDelivPrev).abs();
 for k,z in [('le0.001',er<=.00100001),('le0.002',er<=.00200001),('gt0.002',er>.00200001),('gt0.01',er>.01000001),('gt0.05',er>.05000001),('gt0.1',er>.10000001)]:errhist[k]+=int(z.sum())
 for gr in ['BP','BPFiasp','Control']:
  v=ageok&(grp==gr);group[gr]['adult_raw_rows']+=int(v.sum());group[gr]['adult_rows_rand_to_day91']+=int((v&dt.between(0,91)).sum());group[gr]['adult_rows_pre_rand']+=int((v&(dt<0)).sum());group[gr]['adult_rows_after_day91']+=int((v&(dt>91)).sum());group[gr]['adult_dates_missing']+=int((v&dt.isna()).sum());group[gr]['cgm_40_400_rows']+=int((v&d.CGMVal.between(40,400)).sum())
 meal.update(counts(d.loc[d.MealTimeDose!=0,'MealTimeDose']));cnt['meal_positive_raw_size_derived_missing']+=int(((d.MealTimeDose!=0)&d.MealSize.isna()).sum())
 for col in ['PtWeight','CGMVal','BasalDelivPrev','BolusDelivPrev','MealBolusDelivPrev']:
  if col=='CGMVal':cnt['cgm_39_rows']+=int((d[col]==39).sum());cnt['cgm_401_rows']+=int((d[col]==401).sum())
 z=d[['PtID','UploadIndex','CGMVal','InsDelivAvail','InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolusDelivPrev']].copy();z['t']=t;z['group']=grp;z['adult']=ageok;z['in91']=dt.between(0,91);z['component_ok']=er<=.00200001;parts.append(z)
q=pd.concat(parts,ignore_index=True).sort_values(['PtID','t','UploadIndex']);del parts
dt=q.groupby('PtID').t.diff().dt.total_seconds();di=q.groupby('PtID').UploadIndex.diff();same=(di==1)&((dt-300).abs()<=1);valid=q.CGMVal.between(40,400);prevvalid=q.groupby('PtID').CGMVal.shift().between(40,400);prevavail=q.groupby('PtID').InsDelivAvail.shift().fillna(False).astype(bool)
cnt.update({'time_duplicate_rows':int(q.duplicated(['PtID','t']).sum()),'index_reset_pairs':int((di<0).sum()),'consecutive_step_within1sec':int(same.sum()),'consecutive_step_gap_gt450':int(((di==1)&(dt>450)).sum()),'consecutive_step_nonpositive_time':int(((di==1)&(dt<=0)).sum())})
# Keep availability at previous decision rather than incorrectly judging prior delivery by current availability.
baseok=same&valid&prevvalid&prevavail&q.component_ok&q.adult&q.in91
for gr in ['BP','BPFiasp','Control']:
 mask=baseok&(q.group==gr);group[gr]['adult_91d_basic_event_pair_candidates']=int(mask.sum());group[gr]['candidate_patients']=q.loc[mask,'PtID'].nunique()
 # Max consecutive candidate pair runs. These are diagnostics, not approved transition segments.
 starts=(~mask)|(~mask.shift(fill_value=False))|(q.PtID!=q.PtID.shift());run=starts.cumsum();lens=q.loc[mask].groupby(run[mask]).size();group[gr]['candidate_runs_ge72_pairs']=int((lens>=72).sum());group[gr]['candidate_runs_ge288_pairs']=int((lens>=288).sum());group[gr]['max_candidate_run_pairs']=int(lens.max()) if len(lens) else 0
# Time precision diagnostics date-only is a representational issue; general parser treats midnight but no alignment performed.
ids_stats={}
for id_,df in q.groupby('PtID'):
 ids_stats[str(id_)]={'rows':len(df),'age':int(roster.loc[id_,'AgeAsofEnrollDt']),'group':str(roster.loc[id_,'TrtGroup']),'first_device_date':str(df.t.min()),'last_device_date':str(df.t.max())}
out={'counts':dict(cnt),'component_abs_error_bins':dict(errhist),'meal_time_dose_raw_nonzero':dict(meal),'adult_group_audit':{k:dict(v) for k,v in group.items()},'per_patient':ids_stats,'caution':'Candidates only: no CHO grams, no long-acting/manual injection washout or issue/phase adjudication applied. RCT window is diagnostic calendar RandDt to +91 days; not final source phase extraction.'};dump('iobp2_supplement.json',out)
