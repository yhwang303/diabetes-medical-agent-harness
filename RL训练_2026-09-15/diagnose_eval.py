import torch,json
from model import PatientModel
from data import Data,tensor,ROOT
from train import evaluate
import numpy as np
torch.set_num_threads(4)
m=PatientModel().cuda();ck=torch.load(ROOT/'results/A01_paper_absolute/best.pt');m.load_state_dict(ck['model']);m.eval();D=Data('validation');pid=next(iter(D.patients));b=tensor(D.batch(pid,np.arange(min(64,len(D.patients[pid]['starts'])))))
with torch.no_grad():a=m.rollout(b['state'],b['action'])[0]
with torch.enable_grad():c=m.rollout(b['state'],b['action'])[0]
print(json.dumps({'checkpoint_step':ck['step'],'no_grad_finite':bool(torch.isfinite(a).all()),'grad_enabled_finite':bool(torch.isfinite(c).all()),'max_difference':float((a-c).abs().max())}),flush=True)
