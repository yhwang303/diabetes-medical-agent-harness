"""Isolate reward consistency from response structure, without optimizer updates."""
import copy,hashlib,json
import numpy as np
import torch
from ditr_model import DITRAgent,status_score
from ditr_data import ROOT,Data,tensor

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 source=ROOT/'results/H02_prefix_sim_factual/policy_last.pt';ck=torch.load(source,map_location='cpu');a=DITRAgent(**ck['config']['model']).cuda().eval();a.load_state_dict(ck['agent']);cfg=copy.deepcopy(ck['config']);cfg['name']='C02_H02_point_reward';cfg['model']['point_reward']=True;cfg['adaptations']=['No optimizer updates; exact H02 patient/policy weights; only reward equals unchanged status_score(predicted_glucose)/12.', 'This point forecast reward is not uncertainty-calibrated expected risk.'];b=DITRAgent(**cfg['model']).cuda().eval();b.load_state_dict(ck['agent'])
 data=Data('validation');pid=next(iter(data.patients));batch=tensor(data.batch(pid,np.arange(4),48));history=batch['state'];actions=batch['action']
 with torch.no_grad():
  old=a.patient.rollout(history,actions);new=b.patient.rollout(history,actions)
  for key in ['glucose','states','wtr','memory','initial_value']:torch.testing.assert_close(old[key],new[key],atol=0,rtol=0)
  torch.testing.assert_close(new['reward'],status_score(old['glucose'])/12,atol=0,rtol=0)
  memory=b.patient.encode(history);z=memory[:,-1];features=[];rewards=[]
  for i in range(48):z,reward,features=b.patient.step(z,actions[:,i],memory,features);rewards.append(reward)
  torch.testing.assert_close(torch.stack(rewards,1),new['reward'],atol=2e-5,rtol=2e-5)
 folder=ROOT/'results'/cfg['name'];folder.mkdir(exist_ok=False);ck['config']=cfg;ck['stage']='patient';ck['step']=0;ck['samples']=0;ck.pop('optimizer',None);torch.save(ck,folder/'patient_composed.pt')
 manifest={'status':'composition_complete_no_training','source_checkpoint':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((folder/'patient_composed.pt').read_bytes()).hexdigest(),'inherited_policy_steps':6580,'composition_optimizer_updates':0,'all_weights_exact_H02':True,'only_change':'reward calculation from the same predicted glucose, original status shape','prediction_state_and_initial_policy_exact':True,'step_prefix_reward_parity':True,'config':cfg}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'checks/C02_point_reward_mechanics.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='config'}))
if __name__=='__main__':main()
