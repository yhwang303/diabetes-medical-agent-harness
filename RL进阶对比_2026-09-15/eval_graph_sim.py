"""Factual glucose forecast transfer on independent dev simulator traces; no policy effect claim."""
import json,argparse,hashlib
from pathlib import Path
import torch
import numpy as np
from observable_history import History
from graph_model import GraphPatient
from rl_data import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();folder=ROOT/'results'/args.run;ck=torch.load(folder/'best.pt',map_location='cpu');model=GraphPatient(ck['config']);model.load_state_dict(ck['model']);model.eval();torch.set_num_threads(2);norm=json.loads((ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text());mean=norm['cgm_mmol_l']['mean'];scale=norm['cgm_mmol_l']['scale'];rows=[]
 for path in sorted((ROOT/'results/D01_pilot50k_dev').glob('nominal_p*.json')):
  trace=json.loads(path.read_text());history=History();states=[];features=[];records=trace['records']
  # Original first observation is not recorded, so start at minute5 and wait for 72 real rows.
  for record in records:
   history.append(record['minute'],record['cgm_mg_dl'],record['delivered_basal_u_h']/12,record['bolus_u'] if record['bolus_u']>0 else None,record['meal_g'] if record['meal_g']>0 else None);features.append(history.rows[-1])
   states.append(history.state() if len(history.rows)>=72 else None)
  # First event-age initially unknown is a documented extra transfer boundary; starts well after warmup.
  indices=list(range(144,len(records)-12,12));x=torch.from_numpy(np.stack([states[i] for i in indices]));events=torch.from_numpy(np.stack([np.array(features[i+1:i+13])[:,[2,7,3,8,4,9]] for i in indices]));actions=torch.tensor([[records[j]['delivered_basal_u_h'] for j in range(i+1,i+13)] for i in indices],dtype=torch.float);target=np.array([[records[j]['cgm_mg_dl']/18 for j in range(i+1,i+13)] for i in indices]);last=np.array([records[i]['cgm_mg_dl']/18 for i in indices])[:,None]
  with torch.no_grad():pred=model(x,actions,events).numpy()*scale+mean
  rows.append({'trace':path.name,'patient':trace['patient'],'n':len(indices),'horizons':{str((k+1)*5):{'rmse_mmol_l':float(np.sqrt(((pred[:,k]-target[:,k])**2).mean())),'last_rmse_mmol_l':float(np.sqrt(((last[:,0]-target[:,k])**2).mean()))} for k in [0,2,5,11]}})
 result={'run':args.run,'checkpoint_step':ck['step'],'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'scope':'Independent dev simulator CGM factual forecast given actual future delivered basal and recorded joint events; not causal or closed-loop efficacy','clinical_ready':False};(folder/'sim_factual_forecast.json').write_text(json.dumps(result,indent=2));print(json.dumps({'run':args.run,'traces':len(rows),'rmse60_mean':np.mean([x['horizons']['60']['rmse_mmol_l'] for x in rows]),'last_rmse60_mean':np.mean([x['horizons']['60']['last_rmse_mmol_l'] for x in rows])}))
if __name__=='__main__':main()
