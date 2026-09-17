from pathlib import Path
import pandas as pd,numpy as np,pyarrow as pa,pyarrow.parquet as pq
import json,hashlib,heapq,os,time,concurrent.futures
P=Path(__file__).resolve().parents[1]; TMP=P/'内部'; STEP=300.; HISTORY=21600.
for d in ['患者数据','患者资料','事件记录','隔离记录','审计']: (P/d).mkdir(exist_ok=True)
def save(df,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(path,index=False,compression='zstd')
def load(kind,pid):
 files=list((TMP/'按患者原事件'/kind).glob(f'*/pid={pid}/*.parquet'))
 if not files:return pd.DataFrame()
 return pd.concat([pq.ParquetFile(f).read().to_pandas() for f in files],ignore_index=True)
def nums(d,cols):
 for k in cols:d[k]=pd.to_numeric(d[k],errors='coerce')
def sec(t):return pd.to_datetime(t).astype('datetime64[ns]').astype('int64').to_numpy()/1e9
def refs(d):
 r=d['source_file'].astype(str)+'#RecID='+d['RecID'].astype(str)
 if 'source_record_number' in d:r=r+'#record='+d.source_record_number.astype(str)
 return r
def collapse(d,keys):
 if d.empty:return d,0
 d=d.sort_values(['source_file','RecID'],kind='stable').copy();d['source_ref']=refs(d)
 d['equivalent_raw_count']=d.groupby(keys,dropna=False)['RecID'].transform('size')
 u=d.drop_duplicates(keys).copy();return u,len(d)-len(u)
def segments(starts,ends,rates,invalid):
 """Union equal-rate overlapping delivery; quarantine differing rates, never sum them."""
 if not len(starts):return pd.DataFrame(columns=['start_s','end_s','rate_u_h','status'])
 ix=np.argsort(starts,kind='stable');starts=starts[ix];ends=ends[ix];rates=rates[ix];invalid=invalid[ix]
 points=np.unique(np.r_[starts,ends]); bad_delta=np.zeros(len(points)+1,dtype=int)
 np.add.at(bad_delta,np.searchsorted(points,starts[invalid]),1);np.add.at(bad_delta,np.searchsorted(points,ends[invalid]),-1);bad=np.cumsum(bad_delta)
 lo=[];hi=[];j=0;out=[]
 for k,t in enumerate(points[:-1]):
  while j<len(starts) and starts[j]<=t:
   if not invalid[j]:heapq.heappush(lo,(rates[j],ends[j]));heapq.heappush(hi,(-rates[j],ends[j]))
   j+=1
  while lo and lo[0][1]<=t:heapq.heappop(lo)
  while hi and hi[0][1]<=t:heapq.heappop(hi)
  rate=np.nan
  if bad[k]:status='invalid_event'
  elif not lo:status='gap'
  elif lo[0][0]!=-hi[0][0]:status='rate_conflict'
  else:status='known';rate=lo[0][0]
  end=points[k+1]
  if out and out[-1][3]==status and (status!='known' or out[-1][2]==rate):out[-1][1]=end
  else:out.append([t,end,rate,status])
 return pd.DataFrame(out,columns=['start_s','end_s','rate_u_h','status'])
def integrals(seg,grid,kind):
 if seg.empty:return np.zeros(len(grid))
 s=seg.start_s.to_numpy();e=seg.end_s.to_numpy()
 if kind=='dose':v=np.where(seg.status.eq('known'),seg.rate_u_h.fillna(0).to_numpy()/3600,0)
 else:v=seg.status.eq(kind).to_numpy(dtype=float)
 pref=np.r_[0,np.cumsum((e-s)*v)]
 def f(t):
  ix=np.searchsorted(s,t,side='right')-1;ok=ix>=0;z=np.zeros(len(t));ii=ix[ok];z[ok]=pref[ii]+np.clip(t[ok]-s[ii],0,e[ii]-s[ii])*v[ii];return z
 return f(grid+STEP)-f(grid)
def rolling_all(a,n=72):
 x=np.r_[0,np.cumsum(~a)];r=np.zeros(len(a),bool)
 if len(a)>=n:r[n:]=(x[n:-1]-x[:-n-1])==0 # previous n bins, excludes current
 return r
def bin_events(d,grid,value=None,closed_right=True):
 n=len(grid);counts=np.zeros(n,dtype=np.int32);sums=np.full(n,np.nan)
 if d.empty:return counts,sums
 tt=sec(d.event_time);target=(np.ceil(tt/STEP)*STEP if closed_right else np.floor(tt/STEP)*STEP)
 idx=np.searchsorted(grid,target);ok=(idx>=0)&(idx<n);idx=idx[ok]
 np.add.at(counts,idx,1)
 if value is not None:
  valid=np.isfinite(d[value].to_numpy(dtype=float))[ok];sum0=np.zeros(n);cnt=np.zeros(n,dtype=int)
  np.add.at(sum0,idx[valid],d[value].to_numpy(dtype=float)[ok][valid]);np.add.at(cnt,idx[valid],1);sums[cnt>0]=sum0[cnt>0]
 return counts,sums

def build_one(profile):
 pid=profile['pid'];idstr=f'LOOP_{pid:06d}';starttime=time.time();data={};dedup={};unlocated=False;quarantined=0
 eventdir=P/'事件记录'/idstr;eventdir.mkdir(exist_ok=True)
 # Keep identity/timestamp failures, not silently skipped; known time invalid insulin will block intervals below.
 for kind in ['cgm','basal','bolus','food','exercise','wizard','bgm']:
  d=load(kind,pid)
  if not d.empty:
   assert pd.to_numeric(d.PtID).eq(pid).all()
   bad=d.identity_status.ne('ok')|d.event_time.isna()
   if bad.any():save(d[bad],P/'隔离记录'/idstr/(kind+'_identity_time.parquet'));quarantined+=int(bad.sum())
   if kind in ('basal','bolus') and d.event_time.isna().any():unlocated=True
   # Located invalid basal/bolus remain to invalidate dose coverage; CGM invalid also remains as invalid observations.
   d=d[d.event_time.notna()].copy();d['source_ref']=refs(d)
  data[kind]=d
 c=data['cgm']
 if c.empty:
  save(pd.DataFrame({'patient_id':pd.Series(dtype='str'),'timestamp_utc':pd.Series(dtype='datetime64[ns]')}),P/'患者数据'/(idstr+'.parquet'))
  return {'patient_id':idstr,'pid':pid,'split':profile['split'],'rows':0,'rl_transitions':0,'rl_snapshots':0,'m_candidates':0,'quarantined_identity_time':quarantined,'duplicates_removed':{},'reason':'no_cgm'}
 nums(c,['CGMVal']);cal=c[c.RecordType.ne('CGM')].copy();c=c[c.RecordType.eq('CGM')].copy()
 if not cal.empty:save(cal,eventdir/'calibration.parquet')
 c,nd=collapse(c,['event_time','CGMVal','Units','RecordType']);dedup['cgm']=nd
 if c.empty:raise RuntimeError(f'Patient {pid} has no CGM type')
 c['same_time_conflict']=c.groupby('event_time',dropna=False)['CGMVal'].transform('nunique').gt(1)
 c['valid']=c.identity_status.eq('ok')&c.Units.eq('mmol/L')&np.isfinite(c.CGMVal)&c.CGMVal.gt(0)&~c.same_time_conflict
 save(c,eventdir/'cgm.parquet')
 # A conflicted exact timestamp invalidates that timestamp, even when another origin matches one value.
 cg=c.sort_values(['event_time','source_ref']).drop_duplicates('event_time',keep='last').copy()
 cg['time_s']=sec(cg.event_time);cg['grid_s']=np.ceil(cg.time_s/STEP)*STEP
 cg['bin_readings']=cg.groupby('grid_s').time_s.transform('size')
 cg=cg.sort_values('time_s').drop_duplicates('grid_s',keep='last').sort_values('grid_s')
 grid=np.arange(cg.grid_s.min(),cg.grid_s.max()+STEP,STEP);n=len(grid);ix=np.searchsorted(grid,cg.grid_s)
 out=pd.DataFrame({'patient_id':idstr,'timestamp_utc':pd.to_datetime(grid,unit='s'),'split':profile['split']})
 g=np.full(n,np.nan);mask=np.zeros(n,bool);phys=np.zeros(n,bool);conf=np.zeros(n,bool);obs=np.full(n,np.nan);nr=np.zeros(n,np.int32);ref=np.full(n,None,object)
 good=cg.valid.to_numpy();g[ix[good]]=cg.CGMVal.to_numpy()[good];mask[ix]=good;phys[ix]=True;conf[ix]=cg.same_time_conflict;obs[ix]=cg.time_s;nr[ix]=cg.bin_readings;ref[ix]=cg.source_ref
 out['cgm_mmol_l']=g;out['cgm_mg_dl']=g*18.;out['cgm_mask']=mask;out['cgm_physical_present']=phys;out['cgm_conflict']=conf;out['cgm_observed_time']=pd.to_datetime(obs,unit='s');out['cgm_source_ref']=ref;out['cgm_readings_in_bin']=nr;out['cgm_age_s']=grid-obs
 validix=np.flatnonzero(mask);lastidx=np.maximum.accumulate(np.where(mask,np.arange(n),-1));lastt=np.where(lastidx>=0,obs[np.maximum(lastidx,0)],np.nan)
 out['seconds_since_valid_cgm']=grid-lastt
 segid=np.zeros(n,np.int64);segstart=np.full(n,np.nan)
 if len(validix):
  new=np.r_[True,np.diff(obs[validix])>3600];ids=np.cumsum(new);starts=np.maximum.accumulate(np.where(new,obs[validix],-np.inf));mapidx=np.searchsorted(validix,np.arange(n),side='right')-1;ok=(mapidx>=0)&((grid-lastt)<=3600)
  segid[ok]=ids[mapidx[ok]];segstart[ok]=starts[mapidx[ok]]
 out['cgm_segment_id']=segid
 out['cgm_history_24h_gap_le1h']=(segid>0)&((grid-segstart)>=86400)
 # Basal delivered intervals. Exact rate equality is required; no rate averaging to resolve conflicts.
 b=data['basal'];seg=segments(np.array([]),np.array([]),np.array([]),np.array([],bool))
 if not b.empty:
  nums(b,['Rate','Duration']);b,nd=collapse(b,['event_time','Rate','Duration','BasalType']);dedup['basal']=nd
  b['resolved_rate_u_h']=b.Rate;pause=b.BasalType.eq('suspend')&b.Rate.isna();b.loc[pause,'resolved_rate_u_h']=0.;b['rate_zero_from_suspend_semantics']=pause
  b['valid']=b.identity_status.eq('ok')&b.BasalType.isin(['temp','scheduled','automated','suspend'])&np.isfinite(b.resolved_rate_u_h)&b.resolved_rate_u_h.ge(0)&b.Duration.gt(0)&b.Duration.le(86400000)
  b.loc[b.BasalType.eq('suspend')&b.resolved_rate_u_h.ne(0),'valid']=False
  save(b,eventdir/'basal.parquet');bb=b[b.Duration.gt(0)&np.isfinite(b.Duration)]
  ss=sec(bb.event_time);ee=ss+bb.Duration.to_numpy()/1000.;seg=segments(ss,ee,bb.resolved_rate_u_h.to_numpy(),~bb.valid.to_numpy())
  # A positive-time basal event with unknown duration cannot have a safely bounded pharmacological effect.
  if (b.Duration.isna()|b.Duration.lt(0)).any():unlocated=True
 save(seg,eventdir/'basal_segments.parquet')
 cover=integrals(seg,grid,'known');bc=integrals(seg,grid,'rate_conflict');bi=integrals(seg,grid,'invalid_event');known_u=integrals(seg,grid,'dose')
 complete=np.isclose(cover,STEP,rtol=0,atol=1e-6)&(bc<1e-8)&(bi<1e-8)
 rate=np.full(n,np.nan);held=np.zeros(n,bool)
 if not seg.empty:
  si=np.searchsorted(seg.start_s.to_numpy(),grid,side='right')-1;ok=si>=0;ji=np.maximum(si,0);ok &= (grid<seg.end_s.to_numpy()[ji]) & seg.status.eq('known').to_numpy()[ji]
  rate[ok]=seg.rate_u_h.to_numpy()[ji[ok]];held[ok]=seg.end_s.to_numpy()[ji[ok]]>=grid[ok]+STEP
 out['basal_rate_at_t_u_h']=rate;out['basal_action_u_h']=np.where(complete&held,rate,np.nan);out['basal_u_next_5min']=np.where(complete,known_u,np.nan);out['basal_known_partial_u_next_5min']=known_u;out['basal_coverage_s_next_5min']=cover;out['basal_conflict_s_next_5min']=bc;out['basal_invalid_s_next_5min']=bi
 out['basal_u_prev_5min']=np.r_[np.nan,out.basal_u_next_5min.to_numpy()[:-1]]
 # Bolus: deduplicate identical actual components. Distinct doses at exactly the same time are ambiguous.
 bo=data['bolus'];bol_next=np.zeros(n);bol_prev=np.zeros(n);bol_count=np.zeros(n,np.int32);bol_bad=np.zeros(n,bool);bol_ext_next=np.zeros(n)
 if not bo.empty:
  nums(bo,['Normal','Extended','Duration']);bo,nd=collapse(bo,['event_time','BolusType','Normal','Extended','Duration']);dedup['bolus']=nd
  bo['same_time_conflict']=bo.groupby('event_time').RecID.transform('size').gt(1)
  bo['normal_u']=bo.Normal;bo['extended_u']=bo.Extended
  bo.loc[bo.BolusType.eq('normal')&bo.Extended.isna(),'extended_u']=0.
  bo.loc[bo.BolusType.eq('square')&bo.Normal.isna(),'normal_u']=0.
  bo['duration_s']=np.where(bo.upload_source.eq('Tidepool'),bo.Duration/1000,np.where(bo.upload_source.eq('Diasend'),bo.Duration*60,np.nan))
  bo['valid']=bo.identity_status.eq('ok')&~bo.same_time_conflict&bo.BolusType.isin(['normal','square','dual/square'])&bo.normal_u.ge(0)&bo.extended_u.ge(0)&(bo.extended_u.eq(0)|(bo.duration_s.gt(0)&bo.duration_s.le(86400)))
  save(bo,eventdir/'bolus.parquet')
  valid=bo[bo.valid];bol_count,normal=bin_events(valid,grid,'normal_u',False);bol_next=np.nan_to_num(normal)
  _,normalprev=bin_events(valid,grid,'normal_u',True);bol_prev=np.nan_to_num(normalprev)
  for row in valid[valid.extended_u.gt(0)].itertuples():
   s=row.event_time.timestamp();e=s+row.duration_s;i0=max(0,np.searchsorted(grid,s,side='right')-1);i1=min(n,np.searchsorted(grid,e,side='left'));over=np.maximum(0,np.minimum(grid[i0:i1]+STEP,e)-np.maximum(grid[i0:i1],s));bol_ext_next[i0:i1]+=over/row.duration_s*row.extended_u
  for row in bo[~bo.valid].itertuples():
   s=row.event_time.timestamp();dur=row.duration_s if np.isfinite(row.duration_s) and row.duration_s>0 else STEP;e=s+dur
   i0=max(0,np.searchsorted(grid,s,side='right')-1);i1=min(n,np.searchsorted(grid,e,side='left')+1);bol_bad[i0:i1]=True
 bol_next+=bol_ext_next;bol_prev=np.r_[0,bol_next[:-1]]
 out['bolus_event_count_next_5min']=bol_count;out['bolus_recorded_u_next_5min']=np.where((bol_count>0)|(bol_ext_next>0),bol_next,np.nan);out['bolus_recorded_u_prev_5min']=np.where((np.r_[0,bol_count[:-1]]>0)|(np.r_[0,bol_ext_next[:-1]]>0),bol_prev,np.nan);out['bolus_ambiguous_next_5min']=bol_bad;out['bolus_log_absence_is_unknown']=(bol_count==0)&(bol_ext_next==0)
 # No event is represented as missing, not as known physical zero. Summed logged amount is explicitly conditional.
 out['insulin_logged_total_u_next_5min']=np.where(complete&~bol_bad,known_u+bol_next,np.nan)
 for kind,val,unit in [('food','CarbsNet','CarbUnits'),('exercise',None,None),('wizard','InsulinOnBoard',None),('bgm','BGMVal','Units')]:
  d=data[kind]
  if d.empty:
   dedup[kind]=0
   if kind=='food':out['carbs_recorded_g_prev_5min']=np.nan;out['carb_event_count_prev_5min']=0;out['carbs_ambiguous_prev_5min']=False
   if kind=='exercise':out['exercise_event_count_prev_5min']=0
   if kind=='wizard':out['wizard_iob_u']=np.nan
   continue
  if val:nums(d,[val])
  keys=['event_time']+([val] if val else ['ExerciseName','DurationValue','DurationUnits','DistanceValue','DistanceUnits','EnergyValue','EnergyUnits','ReportedIntensity'])+([unit] if unit else [])
  d,nd=collapse(d,keys);dedup[kind]=nd;d['valid']=d.identity_status.eq('ok')
  if val:d['valid'] &= np.isfinite(d[val])&d[val].ge(0)
  if kind=='food':
   d['same_time_conflict']=d.groupby('event_time').RecID.transform('size').gt(1);d['valid'] &= d.CarbUnits.eq('grams')|d.CarbUnits.eq('g');d['valid'] &= ~d.same_time_conflict
   cnt,su=bin_events(d[d.valid],grid,val,True);badcnt,_=bin_events(d[~d.valid],grid,None,True);out['carbs_recorded_g_prev_5min']=su;out['carb_event_count_prev_5min']=cnt;out['carbs_ambiguous_prev_5min']=badcnt>0
  elif kind=='exercise':out['exercise_event_count_prev_5min']=bin_events(d[d.valid],grid,None,True)[0]
  elif kind=='wizard':
   w=d[d.valid].sort_values('event_time').copy();w['grid']=np.ceil(sec(w.event_time)/STEP)*STEP;w=w.drop_duplicates('grid',keep='last');z=np.full(n,np.nan);ii=np.searchsorted(grid,w.grid);ok=(ii<n)&(ii>=0);z[ii[ok]]=w[val].to_numpy()[ok];out['wizard_iob_u']=z
  save(d,eventdir/(kind+'.parquet'))
 # Structural cohort eligibility, without forcing food/exercise presence or imputing missing modalities.
 hist=rolling_all(complete)&rolling_all(~bol_bad);cg_hist=(segid>0)&((grid-segstart)>=HISTORY)
 study=(grid>=profile['study_start_s'])&(grid+STEP<=profile['study_end_s'])
 state=mask&hist&cg_hist&study&(grid-HISTORY>=profile['study_start_s'])&profile['regimen_eligible']&(not unlocated)
 action=complete&held&np.isfinite(rate)&(rate<=20)&~bol_bad
 snap=state&action
 trans=snap&np.r_[mask[1:],False]&np.r_[segid[1:]==segid[:-1],False]
 # Candidate replay windows only: do not claim meal completeness or physiological identification.
 foodhist=np.r_[0,np.cumsum(out.carb_event_count_prev_5min.to_numpy())];ii=np.maximum(0,np.arange(n)-72);food6h=(foodhist[np.arange(n)+1]-foodhist[ii])>0
 mcan=trans&profile['weight_valid']&food6h
 episode=np.cumsum(trans&~np.r_[False,trans[:-1]]);episode=np.where(trans,episode,0)
 out['basal_bolus_history_6h_ok']=hist;out['cgm_history_6h_gap_le1h']=cg_hist;out['in_study_window']=study;out['rl_snapshot_eligible']=snap;out['rl_transition_eligible']=trans;out['physiology_candidate_only']=mcan;out['episode_id']=episode;out['extreme_basal_rate_gt20']=rate>20
 # Validate no timestamp/identity crossing and no fabricated glucose before writing.
 assert out.patient_id.eq(idstr).all() and out.timestamp_utc.is_unique and out.timestamp_utc.is_monotonic_increasing
 assert np.array_equal(np.isfinite(g),mask)
 assert (obs[mask]<=grid[mask]).all() and (grid[mask]-obs[mask]<STEP+1e-6).all()
 assert np.allclose(out.basal_u_next_5min.to_numpy()[held&complete],rate[held&complete]/12,rtol=1e-7,atol=1e-7)
 assert not trans[-1]
 save(out,P/'患者数据'/(idstr+'.parquet'))
 result={'patient_id':idstr,'pid':pid,'split':profile['split'],'rows':n,'observed_cgm_bins':int(mask.sum()),'cgm_conflict_times':int(c.same_time_conflict.groupby(c.event_time).max().sum()),'basal_full_bins':int(complete.sum()),'basal_conflict_bins':int((bc>0).sum()),'rl_snapshots':int(snap.sum()),'rl_transitions':int(trans.sum()),'episodes':int(episode.max()),'m_candidates':int(mcan.sum()),'food_event_bins':int((out.carb_event_count_prev_5min>0).sum()),'exercise_event_bins':int((out.exercise_event_count_prev_5min>0).sum()),'quarantined_identity_time':quarantined,'unlocatable_insulin_event':bool(unlocated),'duplicates_removed':dedup,'elapsed_s':round(time.time()-starttime,2),'file_bytes':(P/'患者数据'/(idstr+'.parquet')).stat().st_size}
 (P/'审计/逐患者').mkdir(exist_ok=True);(P/'审计/逐患者'/(idstr+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

def safe_json(x):
 if isinstance(x,dict):return {str(k):safe_json(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe_json(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if x is None or (not isinstance(x,(dict,list,str)) and pd.isna(x)):return None
 if isinstance(x,pd.Timestamp):return x.isoformat()
 return x

def prepare_profiles():
 import duckdb
 c=duckdb.connect(str(TMP/'metadata.duckdb'),read_only=True)
 roster=c.execute('select * from PtRoster').fetchdf();surveys=c.execute('select * from Surveys').fetchdf();withdraw=c.execute('select * from LOOPPtFinalStatus').fetchdf()
 for name in ['PtID','AgeAtEnrollment']:roster[name]=pd.to_numeric(roster[name],errors='raise')
 surveys['SubjectID']=pd.to_numeric(surveys.SubjectID);withdraw['PtID']=pd.to_numeric(withdraw.PtID)
 aux={}
 for table in ['SampleResults','gluIndices','adverseEvents','deviceDiscontSurvey','deviceIssues','LOOPContactInteraction','LOOPDeviceUploads']:
  frame=c.execute('SELECT * FROM "'+table+'"').fetchdf();key=next(k for k in frame if k.lower() in ['ptid','subjectid']);frame['_patient_key']=pd.to_numeric(frame[key],errors='raise')
  unknown=frame[~frame._patient_key.isin(roster.PtID)]
  if len(unknown):save(unknown,P/'隔离记录'/(table+'_unknown_patient.parquet'))
  aux[table]={int(pid):g.drop(columns='_patient_key').to_dict('records') for pid,g in frame.groupby('_patient_key') if pid in set(roster.PtID)}
 issue=json.loads((TMP/'IssueRpt_valid.json').read_text());aux['LOOPDeviceIssueRpt']={}
 for r in issue:
  if int(r['PtID']) in set(roster.PtID):
   up=c.execute('SELECT PtID FROM LOOPDeviceUploads WHERE RecID=?',[r['ParentLOOPDeviceUploadsID']]).fetchall()
   r['identity_status']='unknown_upload' if not up else ('ok' if int(up[0][0])==int(r['PtID']) else 'patient_upload_mismatch')
   aux['LOOPDeviceIssueRpt'].setdefault(int(r['PtID']),[]).append(r)
 # Freeze patient splits before any episode/window construction. New F and RL must share these IDs.
 split={}
 for cohort,g in roster.groupby('PtCohort',dropna=False):
  for adult,h in g.groupby(g.AgeAtEnrollment.ge(18)):
   ids=sorted(h.PtID,key=lambda x:hashlib.sha256(f'loop-v1-20260914:{x}'.encode()).hexdigest());N=len(ids)
   for j,pid in enumerate(ids):split[int(pid)]=('train' if j<int(.7*N) else 'validation' if j<int(.85*N) else 'sealed_test' if j<int(.95*N) else 'demo')
 profiles=[]
 for row in roster.to_dict('records'):
  pid=int(row['PtID']);ss=surveys[surveys.SubjectID.eq(pid)];base=ss[ss.Period.eq('Baseline')];reasons=[]
  b=base.iloc[0].to_dict() if len(base)==1 else {}
  def num(k):return pd.to_numeric(b.get(k),errors='coerce')
  risk=False
  for cols,target in [(['insulin_type','insulin_type_f'],5),(['insulin_injection','insulin_injection_f'],1),(['afrezza','afrezza_f'],1),(['pregnant','pregnant_f'],1)]:
   if ss[cols].apply(pd.to_numeric,errors='coerce').eq(target).any().any():risk=True
  if row['AgeAtEnrollment']<18:reasons.append('not_adult_at_enrollment')
  if len(base)!=1:reasons.append('baseline_missing_or_nonunique')
  if num('insulin_type') not in [1,2,3,4]:reasons.append('rapid_insulin_not_documented')
  if num('insulin_injection')!=0:reasons.append('no_extra_injection_not_documented')
  if num('afrezza')!=0:reasons.append('no_afrezza_not_documented')
  if num('gender')!=1 and num('pregnant')!=2:reasons.append('nonpregnancy_not_documented')
  if risk:reasons.append('risk_reported_in_any_survey_cohort_exclusion')
  enroll=pd.to_datetime(row['EnrollDt'],format='%m/%d/%Y',errors='coerce');visit=pd.to_datetime(row['VisitSchedStartDt'],format='%m/%d/%Y',errors='coerce');start=max(enroll,visit) if pd.notna(visit) and pd.notna(enroll) else enroll
  if pd.isna(start):reasons.append('missing_study_start');start=pd.Timestamp('2100-01-01')
  end=start+pd.DateOffset(months=6)
  wd=pd.to_datetime(withdraw.loc[withdraw.PtID.eq(pid),'WithdrawDt'],format='%m/%d/%Y',errors='coerce').dropna()
  if len(wd):end=min(end,wd.min())
  weight=num('weight');weightkg=weight*.45359237 if np.isfinite(weight) else None;weightok=weightkg is not None and 20<=weightkg<=300
  profile={'patient_id':f'LOOP_{pid:06d}','pid':pid,'split':split[pid],'age_at_enrollment':int(row['AgeAtEnrollment']),'cohort':row['PtCohort'],'regimen_eligible':not reasons,'cohort_exclusion_reasons':reasons,'study_start_s':start.timestamp(),'study_end_s':end.timestamp(),'weight_lbs':float(weight) if np.isfinite(weight) else None,'weight_kg':weightkg,'weight_valid':weightok,'baseline':b,'roster':row,'surveys':ss.to_dict('records'),'withdrawals':withdraw[withdraw.PtID.eq(pid)].to_dict('records'),'entry_time_availability':'event timestamps only; meal entry-time and online availability not verified','zero_bolus_assumption':'absence of a recorded event is not verified absence of administration','clinical_ready':False,'physiological_identification_validated':False}
  profile['other_original_tables']={name:by.get(pid,[]) for name,by in aux.items()}
  profile=safe_json(profile)
  profiles.append(profile)
  (P/'患者资料'/(profile['patient_id']+'.json')).write_text(json.dumps(profile,ensure_ascii=False,indent=2,allow_nan=False))
 pd.DataFrame([{'patient_id':p['patient_id'],'split':p['split'],'age_at_enrollment':p['age_at_enrollment'],'cohort':p['cohort']} for p in profiles]).to_csv(P/'审计/patient_split_manifest.csv',index=False)
 (P/'审计/profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,allow_nan=False));c.close();return profiles

if __name__=='__main__':
 profiles=prepare_profiles();results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  fut={pool.submit(build_one,p):p for p in profiles}
  for f in concurrent.futures.as_completed(fut):
   p=fut[f]
   try:r=f.result()
   except Exception as e:
    print('FAILED',p['pid'],repr(e),flush=True)
    for pending in fut:pending.cancel()
    raise
   results.append(r);print('PATIENT_DONE',r['patient_id'],r['rows'],r['rl_transitions'],len(results),flush=True)
 (P/'审计/patient_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('ALL_DONE',len(results),flush=True)
