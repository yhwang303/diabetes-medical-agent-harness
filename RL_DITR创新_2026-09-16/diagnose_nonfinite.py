import json,sys
import numpy as np
import torch
from ditr_model import DITRAgent
from ditr_data import Data,tensor,ROOT

torch.set_num_threads(4)
ck=torch.load(ROOT/'results/R01_reference_paper_lr/patient_002000.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);data=Data('validation');rng=np.random.default_rng(260915)
result={'weights_all_finite':all(torch.isfinite(x).all().item() for x in agent.state_dict().values()),'cases':[]}
with torch.no_grad():
 for pid,d in data.patients.items():
  positions=np.sort(rng.choice(len(d['starts']),min(32,len(d['starts'])),replace=False));b=tensor(data.batch(pid,positions));out=agent.patient.rollout(b['state'],b['action'])
  bad={k:int((~torch.isfinite(v)).sum()) for k,v in out.items() if not torch.isfinite(v).all()}
  if bad:
   case={'patient':pid,'bad_outputs':bad,'inputs_finite':bool(torch.isfinite(b['state']).all()),'modes':{}}
   for mode in ['math_fp32','default_bf16','math_bf16']:
    with torch.backends.cuda.sdp_kernel(enable_flash=mode=='default_bf16',enable_math=True,enable_mem_efficient=mode=='default_bf16'):
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled='bf16' in mode):
      o=agent.patient.rollout(b['state'],b['action']);case['modes'][mode]={k:int((~torch.isfinite(v)).sum()) for k,v in o.items()}
   result['cases'].append(case)
(ROOT/'checks/nonfinite_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
