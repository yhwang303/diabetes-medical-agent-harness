"""Check actual training arrays, time alignment, masking and mixed-loss gradients."""
import copy,json,time
import numpy as np
import torch
from ditr_data import ROOT,Data,tensor
from ditr_model import DITRAgent,status_score
from ditr_losses import patient_loss
from paired_data import PairedData,paired_path_loss

def main():
 torch.set_num_threads(4);torch.manual_seed(260915);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 data=PairedData();raw=data.batch(np.arange(4),np.random.default_rng(260915));b=tensor(raw);real=Data('train');loop=tensor(next(real.batches(256,260915,12)));checks=[]
 assert real.total==1653421 and len(real.patients)==225
 # Future factual records belong only to training targets/consistency states.
 np.testing.assert_array_equal(raw['state'][:,:,:20],data.arrays['features'][:4,:,:72].reshape(20,72,20))
 np.testing.assert_array_equal(raw['final_state'][:,:,:20],data.arrays['features'][:4].reshape(20,120,20)[np.arange(20)[:,None],raw['length'][:,None]+np.arange(72)])
 ideal={'glucose':b['target'].clone(),'reward':status_score(b['target'])/12};zero,_=paired_path_loss(ideal,b);assert float(zero)<1e-10
 shifted={'glucose':ideal['glucose']+3,'reward':ideal['reward']+.02};invariant,_=paired_path_loss(shifted,b);assert float(invariant)<1e-10
 for mode,folder in [('recursive','R03_reference_categorical'),('action_prefix','H01_action_prefix')]:
  torch.cuda.reset_peak_memory_stats();ck=torch.load(ROOT/'results'/folder/'patient_best.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda();agent.load_state_dict(ck['agent']);agent.policy.requires_grad_(False);target=copy.deepcopy(agent.patient).eval().requires_grad_(False);begin=time.time()
  agent.patient.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   real_loss,_=patient_loss(agent.patient,target,loop);agent.patient.eval();fact,_,pred=patient_loss(agent.patient,target,b,return_rollout=True);pair,parts=paired_path_loss(pred,b);loss=real_loss+fact+pair
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(agent.patient.parameters(),5.);assert torch.isfinite(loss) and torch.isfinite(grad)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.patient.decoder.parameters());assert all(p.grad is None for p in agent.policy.parameters());assert all(p.grad is None for p in target.parameters())
  torch.cuda.synchronize();checks.append({'mode':mode,'real_loss':float(real_loss),'sim_factual_loss':float(fact),'pair_loss':float(pair),'gradient_norm':float(grad),'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'seconds':time.time()-begin,'patient_transition_gradient':True,'policy_and_target_frozen':True})
  del agent,target,loss,real_loss,fact,pair,pred,ck;torch.cuda.empty_cache()
 result={'status':'passed','groups':data.total,'arms':data.manifest['arms'],'scenario_count':len(data.manifest['scenarios']),'data_hashes_verified':True,'shared_observed_histories_identical':True,'future_history_index_exact':True,'paired_zero_error':float(zero),'paired_shared_offset_invariance':float(invariant),'loop_patients':225,'loop_origins':1653421,'checks':checks,'not_efficacy_evidence':True}
 (ROOT/'checks/paired_training_check.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
