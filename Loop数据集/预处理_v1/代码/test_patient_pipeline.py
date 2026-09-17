import importlib.util,tempfile
from pathlib import Path
import numpy as np,pandas as pd
s=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('02_build.py'));b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
with tempfile.TemporaryDirectory(dir=b.TMP,prefix='synthetic_e2e_') as td:
 b.P=Path(td)
 for k in ['患者数据','患者资料','事件记录','隔离记录','审计']:(b.P/k).mkdir()
 ts=pd.date_range('2020-01-01',periods=289,freq='5min');base={'PtID':'1','identity_status':'ok','upload_source':'Tidepool'}
 c=pd.DataFrame([{**base,'RecID':str(i),'event_time':t,'UTCDtTm':str(t),'CGMVal':'6.0','Units':'mmol/L','RecordType':'CGM','source_file':'CGM.txt'} for i,t in enumerate(ts)])
 d=pd.DataFrame([{**base,'RecID':'1','event_time':ts[0],'UTCDtTm':str(ts[0]),'Rate':'1.2','Duration':'86400000','BasalType':'scheduled','source_file':'Basal.txt'}])
 bol=pd.DataFrame([{**base,'RecID':'1','event_time':ts[84],'UTCDtTm':str(ts[84]),'Normal':'2','Extended':None,'Duration':None,'BolusType':'normal','source_file':'Bolus.txt'}])
 source={'cgm':c,'basal':d,'bolus':bol};b.load=lambda k,pid:source.get(k,pd.DataFrame()).copy()
 profile={'pid':1,'split':'train','regimen_eligible':True,'study_start_s':ts[0].timestamp(),'study_end_s':ts[-1].timestamp()+300,'weight_valid':True}
 r=b.build_one(profile);a=pd.read_parquet(b.P/'患者数据/LOOP_000001.parquet')
 assert r['rl_transitions']==216,r
 assert a.loc[84,'bolus_recorded_u_next_5min']==2 and pd.isna(a.loc[84,'bolus_recorded_u_prev_5min'])
 assert a.loc[85,'bolus_recorded_u_prev_5min']==2 and pd.isna(a.loc[85,'bolus_recorded_u_next_5min'])
 assert a.cgm_mask.all() and a.carbs_recorded_g_prev_5min.isna().all()
 assert np.allclose(a.basal_u_next_5min.dropna(),.1)
 # Remove a single glucose value and confirm there is an actual NaN, not a forward-fill.
 source['cgm']=c.drop(index=100);r=b.build_one(profile);a=pd.read_parquet(b.P/'患者数据/LOOP_000001.parquet')
 assert not a.loc[100,'cgm_mask'] and pd.isna(a.loc[100,'cgm_mmol_l'])
 assert not a.loc[99,'rl_transition_eligible'] and not a.loc[100,'rl_transition_eligible']
 print('Synthetic end-to-end checks passed: dose conservation, missing CGM, absent optional modalities, bolus boundary, and 216 expected transitions.')
