"""Separate patient-model and policy candidate changes on common observed histories."""
import hashlib,json
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 cases=json.loads((ROOT/'action_probe/samples.json').read_text());h=torch.tensor(np.array([x['history'] for x in cases],dtype='float32'),device='cuda');rows=[]
 for run in ['R03_reference_categorical','H01_action_prefix']:
  patient_digests=[]
  for weight in ['policy_002000.pt','policy_last.pt']:
   ck=torch.load(ROOT/'results'/run/weight,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
   sha=hashlib.sha256()
   for name,value in sorted(agent.patient.state_dict().items()):sha.update(name.encode());sha.update(value.detach().cpu().numpy().tobytes())
   patient_digests.append(sha.hexdigest())
   with torch.no_grad():
    z=agent.patient.encode(h)[:,-1];mu,std=agent.policy.parameters_at(z);candidate=agent.policy.candidates(z);chosen=torch.cat([agent.plan_batch(h[i:i+4])[0] for i in range(0,len(h),4)])
   detail=[{'patient':case['patient'],'minute':case['minute'],'policy_mean_u_h':float(mu[i]),'policy_std':float(std[i]),'initial_min_candidate_u_h':float(candidate[i].min()),'initial_max_candidate_u_h':float(candidate[i].max()),'planned_first_u_h':float(chosen[i]),'probe_reference_basal_u_h':case['arms'][0]['records'][0]['delivered_basal_u_h']} for i,case in enumerate(cases)]
   rows.append({'run':run,'policy_step':ck['step'],'patient_sha256':sha.hexdigest(),'mean_policy_mean_u_h':float(mu.mean()),'mean_policy_std':float(std.mean()),'mean_candidate_max_u_h':float(candidate.max(1).values.mean()),'mean_planned_first_u_h':float(chosen.mean()),'cases':detail});del agent,ck;torch.cuda.empty_cache()
  assert patient_digests[0]==patient_digests[1],'Checkpoint comparison also changed patient model'
 result={'scope':'same28 development probe histories; not independent control results; fixed patient weights within each early/final comparison','rows':rows};(ROOT/'checks/policy_candidate_support.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps([{k:v for k,v in x.items() if k!='cases'} for x in rows]),flush=True)

if __name__=='__main__':main()
