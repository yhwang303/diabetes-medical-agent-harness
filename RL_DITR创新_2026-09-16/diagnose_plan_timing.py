"""Inspect when a frozen planner intends to act; no simulated outcomes or training."""
import argparse,json
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent
from chunk_planning import plan_chunks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 ck=torch.load(ROOT/'results'/args.run/'policy_last.pt',map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 cases=json.loads((ROOT/'action_probe/samples.json').read_text());h=torch.tensor(np.array([x['history'] for x in cases],dtype='float32'),device='cuda');rows=[]
 with torch.no_grad():
  for i in range(0,len(h),2):
   first,details=plan_chunks(agent,h[i:i+2]);paths=np.array(details['planned_actions_u_h'])
   for j,path in enumerate(paths):
    row=cases[i+j];blocks=path.reshape(3,16).mean(1);rows.append({'patient':row['patient'],'minute':row['minute'],'planned_block_rates_u_h':blocks.tolist(),'reference_basal_u_h':row['arms'][0]['records'][0]['delivered_basal_u_h'],'first_rate_u_h':float(first[j])})
 result={'run':args.run,'scope':'28 exposed observed histories; intended action timing only, not efficacy or causal proof','mean_block_rates_u_h':np.mean([r['planned_block_rates_u_h'] for r in rows],axis=0).tolist(),'last_minus_first_u_h':float(np.mean([r['planned_block_rates_u_h'][-1]-r['first_rate_u_h'] for r in rows])),'cases':rows}
 (ROOT/'checks'/(args.run+'_plan_timing.json')).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
