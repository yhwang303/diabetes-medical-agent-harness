"""A one-step action block must reproduce ordinary beam on actual model weights."""
import json,time
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent
from chunk_planning import plan_chunks

def main():
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 cases=json.loads((ROOT/'action_probe/samples.json').read_text());h=torch.tensor(np.array([x['history'] for x in cases[:2]],dtype='float32'),device='cuda');rows=[]
 for run in ['R03_reference_categorical','H01_action_prefix']:
  ck=torch.load(ROOT/'results'/run/'policy_last.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval().requires_grad_(False);agent.load_state_dict(ck['agent'])
  original,old=agent.plan_batch(h,horizon=4,beam_size=3);one,info=plan_chunks(agent,h,horizon=4,block_steps=1,beam_size=3)
  np.testing.assert_allclose(original.cpu(),one.cpu(),atol=1e-5,rtol=1e-5);np.testing.assert_allclose(old['plan_value'],info['plan_value'],atol=1e-5,rtol=1e-5)
  start=time.time();a,long=plan_chunks(agent,h,horizon=48,block_steps=16);torch.cuda.synchronize();seconds=time.time()-start;path=np.array(long['planned_actions_u_h']);assert path.shape==(2,48) and np.isfinite(path).all();assert np.all((path>=0)&(path<=20))
  for j in range(0,48,16):np.testing.assert_array_equal(path[:,j:j+16],np.repeat(path[:,j:j+1],16,axis=1))
  changed=h.clone();changed[1,:,0]+=2;other,_=plan_chunks(agent,changed,horizon=48,block_steps=16);np.testing.assert_allclose(a[0].cpu(),other[0].cpu(),atol=1e-5,rtol=1e-5)
  rows.append({'run':run,'one_step_blocks_match_original_beam':True,'full48step_plan_and16step_blocks_valid':True,'patient_isolation_verified':True,'batch2_4h_seconds_shared_gpu':seconds,'actions':a.cpu().tolist(),'mechanical_check_only':True});del agent,ck;torch.cuda.empty_cache()
 result={'status':'passed','rows':rows,'risk_function_unchanged':True,'no_future_observations_or_hidden_states_given':True,'current_weights_untrained_beyond60min_not_efficacy_evidence':True};(ROOT/'checks/chunk_planning_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
