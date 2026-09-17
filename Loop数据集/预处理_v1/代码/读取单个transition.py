"""Read one already-qualified transition; preserve NaNs. No model training."""
from pathlib import Path
import argparse,pandas as pd,numpy as np
P=Path(__file__).resolve().parents[1]
STATE=['cgm_mmol_l','cgm_mask','cgm_age_s','seconds_since_valid_cgm','basal_u_prev_5min','bolus_recorded_u_prev_5min','carbs_recorded_g_prev_5min','carb_event_count_prev_5min','carbs_ambiguous_prev_5min','exercise_event_count_prev_5min','wizard_iob_u']
def read_transition(patient_id,row_index=None):
 if not patient_id.startswith('LOOP_') or not patient_id[5:].isdigit():raise ValueError('Use LOOP_XXXXXX patient ID')
 d=pd.read_parquet(P/'患者数据'/(patient_id+'.parquet'))
 available=np.flatnonzero(d.rl_transition_eligible.to_numpy())
 if not len(available):raise ValueError('This patient has no qualified transition')
 i=int(available[0]) if row_index is None else int(row_index)
 if i<71 or i+1>=len(d) or not d.iloc[i].rl_transition_eligible:raise ValueError('Not a qualified transition row')
 return {'patient_id':patient_id,'split':d.iloc[i]['split'],'time':d.iloc[i].timestamp_utc,'history':d.iloc[i-71:i+1][['timestamp_utc']+STATE].copy(),'action_basal_u_h':float(d.iloc[i].basal_action_u_h),'action_basal_u_in_5min':float(d.iloc[i].basal_u_next_5min),'next_history':d.iloc[i-70:i+2][['timestamp_utc']+STATE].copy(),'clinical_terminal':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('patient_id');p.add_argument('--row',type=int);a=p.parse_args();x=read_transition(a.patient_id,a.row)
 print({k:v for k,v in x.items() if k not in ['history','next_history']});print('history:',x['history'].shape,'next_history:',x['next_history'].shape,'NaNs retained:',int(x['history'].isna().sum().sum()))
