"""Separate background forecast error from intervention effect error on saved probes."""
import hashlib,json
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent,status_score

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 cases=json.loads((ROOT/'action_probe/samples.json').read_text());rows=[]
 for run,weight in [('H02_prefix_sim_factual','patient_best.pt'),('H02_prefix_sim_factual','patient_last.pt'),('H03_prefix_sim_paired','patient_best.pt'),('C01_response_patient_composition','patient_composed.pt')]:
  path=ROOT/'results'/run/weight;ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False);errors=[];effects=[];decisions=[]
  with torch.no_grad():
   for case in cases:
    assert min(len(a['records']) for a in case['arms'])==48
    actions=torch.tensor([[r['delivered_basal_u_h'] for r in a['records']] for a in case['arms']],dtype=torch.float32,device='cuda');history=torch.tensor(np.repeat(np.array(case['history'])[None],len(actions),axis=0),dtype=torch.float32,device='cuda');truth=np.array([[r['cgm'] for r in a['records']] for a in case['arms']]);bg=torch.tensor([[r['bg']/18 for r in a['records']] for a in case['arms']],device='cuda')
    out=agent.patient.rollout(history,actions);g=out['glucose'].cpu().numpy()*18;errors.append(g[0]-truth[0]);effects.append((g[1:]-g[:1])-(truth[1:]-truth[:1]));discount=torch.pow(torch.tensor(.9**(1/12),device='cuda'),torch.arange(48,device='cuda'));true=(status_score(bg)/12*discount).sum(1);pred=(out['reward']*discount).sum(1);chosen=int(pred.argmax());decisions.append({'agreement':chosen==int(true.argmax()),'regret':float(true.max()-true[chosen])})
  e=np.array(errors);de=np.concatenate(effects);summary={str(h*5):{'reference_bias_mg_dl':float(e[:,h-1].mean()),'reference_rmse_mg_dl':float(np.sqrt((e[:,h-1]**2).mean())),'effect_mae_mg_dl':float(np.abs(de[:,h-1]).mean())} for h in [3,6,12,24,48]}
  rows.append({'run':run,'weight':weight,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'patient_steps':ck.get('step'),'summary':summary,'reward_choice_agreement':float(np.mean([d['agreement'] for d in decisions])),'reward_regret':float(np.mean([d['regret'] for d in decisions]))})
 result={'scope':'exposed development only; diagnosing fixed short-horizon checkpoint selection versus final patient weights, not selecting a winner or rerunning baseline control','histories':len(cases),'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'results':rows};(ROOT/'checks/reference_forecast_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
