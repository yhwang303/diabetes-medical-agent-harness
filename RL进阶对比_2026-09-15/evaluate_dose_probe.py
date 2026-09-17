"""Compare graph action-effect size with independently cloned physiological trajectories."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from graph_model import GraphPatient
from rl_data import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();source=ROOT/'results/independent_dose_probe/samples.json';rows=json.loads(source.read_text());folder=ROOT/'results'/a.run;ck=torch.load(folder/'best.pt',map_location='cpu');m=GraphPatient(ck['config']);m.load_state_dict(ck['model']);m.eval();torch.set_num_threads(2);norm=json.loads((ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json').read_text());scale=norm['cgm_mmol_l']['scale'];records=[]
 for row in rows:
  states=torch.tensor([row['state']]*3,dtype=torch.float);events=torch.tensor([row['events']]*3,dtype=torch.float);actions=torch.tensor([arm['actual_basal_u_h'] for arm in row['arms']],dtype=torch.float)
  with torch.no_grad():pred=m(states,actions,events).numpy()*scale
  for arm in [1,2]:
   p=pred[arm]-pred[0];bg=np.array(row['arms'][arm]['bg_mmol_l'])-row['arms'][0]['bg_mmol_l'];cgm=np.array(row['arms'][arm]['cgm_mmol_l'])-row['arms'][0]['cgm_mmol_l'];records.append({'patient':row['patient'],'minute':row['minute'],'delta_requested_u_h':row['arms'][arm]['delta_requested_u_h'],'predicted_delta_mmol_l':p.tolist(),'simulator_bg_delta_mmol_l':bg.tolist(),'simulator_cgm_delta_mmol_l':cgm.tolist()})
 summary={}
 for delta in [.1,1.]:
  sub=[x for x in records if x['delta_requested_u_h']==delta];p=np.array([x['predicted_delta_mmol_l'] for x in sub]);g=np.array([x['simulator_bg_delta_mmol_l'] for x in sub]);summary[str(delta)]={str((h+1)*5):{'model_median':float(np.median(p[:,h])),'simulator_median':float(np.median(g[:,h])),'effect_rmse_mmol_l':float(np.sqrt(np.mean((p[:,h]-g[:,h])**2))),'model_negative_fraction':float(np.mean(p[:,h]<0)),'simulator_negative_fraction':float(np.mean(g[:,h]<0))} for h in [0,2,5,11]}
 result={'run':a.run,'step':ck['step'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),'summary':summary,'records':records,'scope':'Independent dev-simulator perturbation effects; not clinical effect estimates or final sealed evidence','clinical_ready':False};(folder/'independent_dose_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps({'run':a.run,'summary':summary},indent=2))
if __name__=='__main__':main()
