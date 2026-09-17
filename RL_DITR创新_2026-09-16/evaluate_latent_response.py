"""Independent observed-window latent effects; future records are labels only."""
import hashlib,json,sys
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent
sys.path.insert(0,str(ROOT.parent/'RL进阶对比_2026-09-15'))
from observable_history import History

def observed_future(case,arm,steps):
 h=History();initial=np.asarray(case['history'],dtype='float32');h.rows=list(initial[:,:20].copy())
 # Last-row age/known features recover sufficient History state; ages cap at 24h.
 h.last=np.where(initial[-1,15:20]>.5,case['minute']-60*np.expm1(initial[-1,10:15].astype('float64')),-np.inf)
 np.testing.assert_allclose(h.state(),initial,atol=1e-6)
 states=[]
 for k,r in enumerate(arm['records'],1):
  h.append(case['minute']+5*k,r['cgm'],r['delivered_basal_u_h']/12,r['bolus_u'] if r['bolus_u']>0 else None,r['meal_g'] if r['meal_g']>0 else None)
  if k in steps:states.append(h.state())
 return np.array(states)

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=ROOT/'results/C01_response_patient_composition/patient_composed.pt';ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 steps=[1,6,12,24,48];rows=[]
 with torch.no_grad():
  for folder in ['action_probe','action_probe_dynamic']:
   probe=ROOT/folder/'samples.json';cases=json.loads(probe.read_text());metrics=[]
   for case in cases:
    if any(len(a['records'])<48 for a in case['arms']):continue
    arms=case['arms'];history=torch.tensor(np.repeat(np.array(case['history'])[None],len(arms),axis=0),dtype=torch.float32,device='cuda');actions=torch.tensor([[r['delivered_basal_u_h'] for r in a['records']] for a in arms],dtype=torch.float32,device='cuda')
    future=np.stack([observed_future(case,a,steps) for a in arms]);target=agent.patient.background.encode(torch.tensor(future.reshape(-1,72,22),device='cuda'))[:,-1].reshape(len(arms),len(steps),-1)
    baseline=agent.patient.background.rollout(history,actions)['states'][:,np.array(steps)-1];response=agent.patient.rollout(history,actions)['states'][:,np.array(steps)-1]
    truth=target[1:]-target[:1];base_effect=baseline[1:]-baseline[:1];new_effect=response[1:]-response[:1]
    for j,step in enumerate(steps):
     metrics.append({'patient':case['patient'],'minute':case['minute'],'minutes':step*5,'zero_mse':float(truth[:,j].square().mean()),'H02_effect_mse':float((base_effect[:,j]-truth[:,j]).square().mean()),'response_effect_mse':float((new_effect[:,j]-truth[:,j]).square().mean())})
   summary={str(h*5):{k:float(np.mean([m[k] for m in metrics if m['minutes']==h*5])) for k in ['zero_mse','H02_effect_mse','response_effect_mse']} for h in steps}
   rows.append({'probe':folder,'samples_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'complete_histories':len(metrics)//len(steps),'summary':summary,'cases':metrics})
 result={'status':'completed','checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'scope':'development latent intervention generalization in frozen H02 coordinates; future observation windows are evaluation labels only, not model inputs; not control efficacy','results':rows}
 (ROOT/'checks/response_latent_development.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in row.items() if k!='cases'} for row in rows]),flush=True)

if __name__=='__main__':main()
