"""Independent postprocessing audit. Reads outputs plus canonical source events, never trains."""
from pathlib import Path
import pandas as pd,numpy as np,pyarrow.parquet as pq,json,hashlib,collections
P=Path(__file__).resolve().parents[1];profiles=json.loads((P/'审计/profiles.json').read_text());pm={p['patient_id']:p for p in profiles};results=[];independent=[];proof={'identity_rows_checked':0,'cgm_values_matched_to_real_events':0,'eligible_rows_checked':0,'dose_sample_checks':0};fail=[]
for f in sorted((P/'患者数据').glob('*.parquet')):
 pid=f.stem;p=pm[pid];a=pd.read_parquet(f);r={'patient_id':pid,'split':p['split'],'age_at_enrollment':p['age_at_enrollment'],'regimen_eligible':p['regimen_eligible'],'cohort_exclusion_reasons':';'.join(p['cohort_exclusion_reasons']),'rows':len(a),'rl_transitions':0,'rl_snapshots':0,'m_candidates':0,'food_present':False,'exercise_present':False}
 if not len(a):r['no_rl_reason']='no_cgm';results.append(r);continue
 assert a.patient_id.eq(pid).all();assert a.split.eq(p['split']).all();assert a.timestamp_utc.is_unique and a.timestamp_utc.is_monotonic_increasing
 assert a.timestamp_utc.diff().dropna().eq(pd.Timedelta(minutes=5)).all();proof['identity_rows_checked']+=len(a)
 obs=a[a.cgm_mask].copy();assert a.loc[~a.cgm_mask,'cgm_mmol_l'].isna().all();assert np.allclose(obs.cgm_mg_dl,obs.cgm_mmol_l*18,rtol=0,atol=1e-12)
 assert (obs.cgm_observed_time<=obs.timestamp_utc).all();assert ((obs.timestamp_utc-obs.cgm_observed_time)<pd.Timedelta(minutes=5)).all()
 ev=pd.read_parquet(P/'事件记录'/pid/'cgm.parquet',columns=['PtID','source_ref','CGMVal','event_time','valid'])
 assert pd.to_numeric(ev.PtID).eq(p['pid']).all();assert ev.source_ref.is_unique
 join=obs.merge(ev,left_on='cgm_source_ref',right_on='source_ref',validate='many_to_one',how='left')
 assert len(join)==len(obs) and join.valid.fillna(False).all();assert np.array_equal(join.cgm_mmol_l.to_numpy(),join.CGMVal.to_numpy());assert join.cgm_observed_time.eq(join.event_time).all();proof['cgm_values_matched_to_real_events']+=len(join)
 elig=a[a.rl_transition_eligible];n=len(a);m=a.rl_transition_eligible.to_numpy();nextmask=np.r_[a.cgm_mask.to_numpy()[1:],False]
 assert (a.loc[m,'cgm_mask']&nextmask[m]).all();assert a.loc[m,'basal_bolus_history_6h_ok'].all();assert a.loc[m,'cgm_history_6h_gap_le1h'].all();assert a.loc[m,'in_study_window'].all()
 assert a.loc[m,'basal_action_u_h'].notna().all();assert not a.loc[m,'bolus_ambiguous_next_5min'].any();assert not a.loc[m,'extreme_basal_rate_gt20'].any();assert not m[-1]
 assert np.allclose(a.loc[m,'basal_coverage_s_next_5min'],300.,rtol=0,atol=1e-6);assert np.allclose(a.loc[m,'basal_u_next_5min'],a.loc[m,'basal_action_u_h']/12,rtol=1e-7,atol=1e-7)
 if len(elig):assert p['regimen_eligible'] and p['age_at_enrollment']>=18
 proof['eligible_rows_checked']+=len(elig)
 r.update({'observed_cgm_bins':int(a.cgm_mask.sum()),'rl_transitions':len(elig),'rl_snapshots':int(a.rl_snapshot_eligible.sum()),'m_candidates':int(a.physiology_candidate_only.sum()),'episodes':int(a.episode_id.max()),'food_present':bool((a.carb_event_count_prev_5min.gt(0)&a.in_study_window).any()),'exercise_present':bool((a.exercise_event_count_prev_5min.gt(0)&a.in_study_window).any()),'rl_with_24h_cgm_context':int((a.rl_transition_eligible&a.cgm_history_24h_gap_le1h).sum()),'basal_complete_bins':int(a.basal_u_next_5min.notna().sum()),'basal_conflict_bins':int(a.basal_conflict_s_next_5min.gt(0).sum()),'nonconstant_or_incomplete_action_bins':int(a.basal_action_u_h.isna().sum()),'file_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 # Direct midpoint integration from original event intervals, independent of sweep implementation.
 if len(independent)<40 and (P/'事件记录'/pid/'basal.parquet').exists():
  b=pd.read_parquet(P/'事件记录'/pid/'basal.parquet');b=b[pd.to_numeric(b.Duration,errors='coerce').gt(0)].copy();ss=b.event_time.astype('datetime64[ns]').astype('int64').to_numpy()/1e9;ee=ss+b.Duration.to_numpy()/1000;rate=b.resolved_rate_u_h.to_numpy();valid=b.valid.to_numpy()
  choices=[]
  for sub in [a[a.basal_u_next_5min.notna()],a[a.basal_conflict_s_next_5min.gt(0)],a[a.basal_coverage_s_next_5min.lt(300)]]:
   if len(sub):choices.append(sub.iloc[len(sub)//2])
  for row in choices:
   t=row.timestamp_utc.timestamp();v=(ss<t+300)&(ee>t);cuts=np.unique(np.r_[t,t+300,np.clip(ss[v],t,t+300),np.clip(ee[v],t,t+300)]);dose=cover=conf=0.
   for l,h in zip(cuts[:-1],cuts[1:]):
    active=v&(ss<(l+h)/2)&(ee>(l+h)/2)
    if not active.any() or (~valid&active).any():continue
    u=np.unique(rate[active])
    if len(u)>1:conf+=h-l
    else:cover+=h-l;dose+=(h-l)*u[0]/3600
   assert abs(dose-row.basal_known_partial_u_next_5min)<1e-5,(pid,t,dose,row.basal_known_partial_u_next_5min)
   assert abs(cover-row.basal_coverage_s_next_5min)<1e-5;assert abs(conf-row.basal_conflict_s_next_5min)<1e-5;proof['dose_sample_checks']+=1
  independent.append(pid)
 r['study_rows']=int(a.in_study_window.sum())
 r['study_rows_with_cgm']=int((a.in_study_window&a.cgm_mask).sum())
 r['study_rows_with_complete_insulin_history']=int((a.in_study_window&a.basal_bolus_history_6h_ok).sum())
 r['study_rows_with_held_action']=int((a.in_study_window&a.basal_action_u_h.notna()).sum())
 r['no_rl_reason']='' if r['rl_transitions'] else (r['cohort_exclusion_reasons'] or 'no_transition_passes_all_structural_rules')
 results.append(r)
 if len(results)%50==0:print('VERIFIED',len(results),flush=True)
assert len(results)==len(pm)==919
ix=pd.DataFrame(results);ix.to_csv(P/'患者索引.csv',index=False)
training=ix[ix.rl_transitions.gt(0)].copy();training.to_csv(P/'审计/rl_patient_manifest.csv',index=False)
# A compact row-range manifest names only eligible transitions, without duplicating patient time-series data.
chunks=[]
for row in training.itertuples():
 a=pd.read_parquet(P/'患者数据'/(row.patient_id+'.parquet'),columns=['timestamp_utc','rl_transition_eligible','episode_id']);a['row_index']=np.arange(len(a));a=a[a.rl_transition_eligible]
 for eid,g in a.groupby('episode_id'):
  chunks.append({'patient_id':row.patient_id,'split':row.split,'episode_id':int(eid),'first_row':int(g.row_index.iloc[0]),'last_transition_row':int(g.row_index.iloc[-1]),'next_state_row':int(g.row_index.iloc[-1]+1),'transitions':len(g),'clinical_terminal':False,'end_reason':'data_eligibility_boundary','start_time':str(g.timestamp_utc.iloc[0]),'last_transition_time':str(g.timestamp_utc.iloc[-1])})
episode_df=pd.DataFrame(chunks)
episode_stats={'count':len(episode_df),'length_transitions_quantiles':episode_df.transitions.quantile([.5,.9,.99,1]).to_dict() if len(episode_df) else {},'at_least_1h':int(episode_df.transitions.ge(12).sum()) if len(episode_df) else 0,'at_least_6h':int(episode_df.transitions.ge(72).sum()) if len(episode_df) else 0,'at_least_24h':int(episode_df.transitions.ge(288).sum()) if len(episode_df) else 0}
episode_df.to_parquet(P/'审计/rl_episode_manifest.parquet',index=False,compression='zstd')
optional={}
for name,cond in [('no_optional_requirement',pd.Series(True,index=training.index)),('require_any_food',training.food_present),('require_any_exercise',training.exercise_present),('require_food_and_exercise',training.food_present&training.exercise_present)]:
 x=training[cond];optional[name]={'patients':len(x),'transitions':int(x.rl_transitions.sum())}
ing=json.loads((P/'审计/ingestion.json').read_text());counts=collections.Counter()
for x in ing.values():counts.update(x['identity_status_counts'])
summary={'patients_archived':919,'patients_with_cgm':int(ix.rows.gt(0).sum()),'grid_rows':int(ix.rows.sum()),'observed_cgm_bins':int(ix.observed_cgm_bins.sum()),'rl_patients':len(training),'rl_transitions':int(training.rl_transitions.sum()),'rl_exposure_hours':float(training.rl_transitions.sum()*5/60),'rl_episodes':len(chunks),'rl_snapshots':int(ix.rl_snapshots.sum()),'physiology_candidate_rows_only':int(ix.m_candidates.sum()),'physiology_validated_patients':0,'episode_statistics':episode_stats,'regimen_eligible_patients':sum(p['regimen_eligible'] for p in profiles),'by_split':training.groupby('split').agg(patients=('patient_id','size'),transitions=('rl_transitions','sum')).to_dict('index'),'optional_modality_sensitivity':optional,'input_identity_status_counts':dict(counts),'proof':proof,'audit_scope':'Retrospective conditional offline research eligibility; no model training, counterfactual validation or clinical readiness. No values imputed.','failures':fail}
(P/'审计/final_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
# Readable preview from a TRAIN patient only. Never choose a held-out subject for demo.
if len(training[training.split.eq('train')]):
 pp=training[training.split.eq('train')].sort_values('rl_transitions',ascending=False).iloc[0].patient_id;a=pd.read_parquet(P/'患者数据'/(pp+'.parquet'));i=int(np.flatnonzero(a.rl_transition_eligible)[0]);a.iloc[max(0,i-12):i+36].to_csv(P/'患者表预览_训练患者.csv',index=False);(P/'审计/preview_patient.txt').write_text(pp)
