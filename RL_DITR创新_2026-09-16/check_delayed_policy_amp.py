"""Disposable AMP integration check; never saves or reuses smoke-test weights."""
import copy,json
import torch
from ditr_data import ROOT,Data,tensor
from ditr_model import DITRAgent
from delayed_policy import delayed_trajectory_loss

def main():
 torch.set_num_threads(4);data=Data('train');b=tensor(next(data.batches(256,260915,12)));rows=[]
 for code in ['H05','R08']:
  cfg=json.loads((ROOT/'configs'/(code+'.json')).read_text());ck=torch.load(ROOT/cfg['initialize_checkpoint'],map_location='cpu');a=DITRAgent(**cfg['model']).cuda();a.load_state_dict(ck['agent']);a.patient.requires_grad_(False);a.patient.eval();ref=copy.deepcopy(a.policy).eval();ref.requires_grad_(False);opt=torch.optim.Adam(a.policy.parameters(),lr=cfg['policy']['lr'],weight_decay=cfg['weight_decay']);torch.manual_seed(260915);metrics=[]
  for i in range(3):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=delayed_trajectory_loss(a,ref,b['state'],**cfg['delayed_policy'])
   assert loss.requires_grad and torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(a.policy.parameters(),5.);assert torch.isfinite(norm);assert all(p.grad is None for p in a.patient.parameters());assert all(p.grad is None for p in ref.parameters());opt.step();metrics.append({'loss':float(loss),'gradient':float(norm),**{k:float(v) for k,v in parts.items()}})
  rows.append({'code':code,'batch':len(b['state']),'discarded_updates':3,'metrics':metrics})
 result={'status':'passed','scope':'disposable three-update AMP integration check; no weights reused in training','initial_failure':'Torch2.0 autocast reused no-grad actor weight casts from first sampling forward; loss lacked grad_fn. Fixed by sampling with grad enabled then detach, matching existing DITRAgent.imagine.','rows':rows};(ROOT/'checks/delayed_policy_amp_integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
