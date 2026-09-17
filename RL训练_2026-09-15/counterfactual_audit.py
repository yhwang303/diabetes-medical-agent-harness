"""Small/local dose perturbation and patient-support scans; diagnostic, not causal validation."""
import argparse,json
import numpy as np
import torch
from data import Data,tensor,ROOT
from model import PatientModel
from train import write_json
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);arg=ap.parse_args();out=ROOT/'results'/arg.run;ck=torch.load(out/'best.pt',map_location='cuda');cfg=ck['config'];m=PatientModel(**cfg['model']).cuda().eval();m.load_state_dict(ck['model']);assert all(torch.equal(v,m.state_dict()[k]) for k,v in ck['model'].items());torch.set_num_threads(4);data=Data('validation');rng=np.random.default_rng(260916);records=[]
 for pid,d in data.patients.items():
  supported=np.flatnonzero(d['length']>=12)
  if not len(supported):continue
  positions=rng.choice(supported,min(32,len(supported)),replace=False);b=tensor(data.batch(pid,positions));n=len(positions);a=b['action'][:,:1]
  # Initial observed rate itself, then a small positive change; no fabricated physiology.
  low=a.clamp(0,19.9);high=low+.1
  baseline=m.rollout(b['state'],low.expand(-1,12))[0]
  raised=m.rollout(b['state'],high.expand(-1,12))[0]
  delta=(raised-baseline).cpu().numpy()
  for j in range(n):records.append({'patient':pid,'row':int(b['row'][j]),'starting_rate_u_h':float(a[j]),'step_plus_0_1_u_h_delta5':float(delta[j,0]),'delta30':float(delta[j,5]),'delta60':float(delta[j,11])})
 vals=np.array([r['delta60'] for r in records]);assert np.isfinite(vals).all()
 result={'weights_loaded_and_exactly_verified':True,'run':arg.run,'checkpoint_step':ck['step'],'n':len(vals),'patients':len(set(r['patient'] for r in records)),'selection':'validation starts with >=60min continuous logged eligibility, fixed seed, max32 per patient','perturbation_u_h':.1,'fraction_negative_60':float((vals<0).mean()),'median_delta60_mmol_l':float(np.median(vals)),'mean_delta60_mmol_l':float(vals.mean()),'max_abs_delta60':float(np.abs(vals).max()),'records':records,'clinical_ready':False,'caveat':'Logged eligibility does not establish support of both alternative actions or absence of unmeasured confounding. No clinical effect estimate.'}
 write_json(out/'local_dose_response.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
