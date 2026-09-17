"""Post-training input ablations on identical validation starts; no retraining or causal claim."""
import argparse, json, hashlib
import numpy as np
import torch
from data import Data, tensor, ROOT
from model import PatientModel
from train import metrics, write_json

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
 torch.set_num_threads(4);out=ROOT/'results'/args.run;path=out/'best.pt'
 ck=torch.load(path,map_location='cuda');m=PatientModel(**ck['config']['model']).cuda().eval()
 m.load_state_dict(ck['model']);assert all(torch.equal(v,m.state_dict()[k]) for k,v in ck['model'].items())
 data=Data('validation');rng=np.random.default_rng(260917);rows=[]
 variants=['logged_actions','hold_initial_rate','mask_food_exercise_history']
 for pid,d in data.patients.items():
  pool=np.flatnonzero(d['length']>=12)
  if not len(pool):continue
  pos=rng.choice(pool,min(32,len(pool)),replace=False);b=tensor(data.batch(pid,pos));pred={}
  for variant in variants:
   state=b['state'].clone();action=b['action']
   if variant=='hold_initial_rate':action=action[:,:1].expand(-1,12)
   if variant=='mask_food_exercise_history':
    # All four fields for each of the two channels: value, observed, age, age-known.
    state[:,:,[3,4,8,9,13,14,18,19]]=0
   pred[variant]=m.rollout(state,action)[0].float().cpu().numpy()
   assert np.isfinite(pred[variant]).all()
  target=b['target'].cpu().numpy();state=b['state'].cpu().numpy()
  for i in range(len(pos)):
   rows.append({'patient':pid,'row':int(b['row'][i]),'has_food_record':bool((state[i,:,8]>.5).any()),'has_exercise_record':bool((state[i,:,9]>.5).any()),'target':target[i].tolist(),'predictions':{k:v[i].tolist() for k,v in pred.items()}})
 summary={}
 for group in ['all','food_record','exercise_record','either_record']:
  selected=[r for r in rows if group=='all' or (group=='food_record' and r['has_food_record']) or (group=='exercise_record' and r['has_exercise_record']) or (group=='either_record' and (r['has_food_record'] or r['has_exercise_record']))]
  summary[group]={str(h*5):{v:metrics([r['target'][h-1] for r in selected],[r['predictions'][v][h-1] for r in selected]) for v in variants} for h in [6,12]} if selected else {}
 result={'run':args.run,'checkpoint_step':ck['step'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weights_loaded_and_exactly_verified':True,'seed':260917,'selection':'up to32 uniformly sampled starts per validation patient with 12 continuous eligible steps; identical starts across variants','patients':len(set(r['patient'] for r in rows)),'samples':len(rows),'summary':summary,'records':rows,'caveat':'Post-training input deletion/plan substitution can be out of distribution. This is not a retrained feature ablation, causal effect, or independent policy evaluation. Logged labels are retained for all variants; held initial actions need not match observed treatment.'}
 write_json(out/'input_diagnostics.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
