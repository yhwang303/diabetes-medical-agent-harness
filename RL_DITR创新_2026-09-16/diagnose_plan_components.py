"""Decompose learned planning scores on saved development interventions only."""
import argparse,hashlib,json
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent,status_score

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--weight',default='patient_best.pt');args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=ROOT/'results'/args.run/args.weight;ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False);rows=[]
 for case in json.loads((ROOT/'action_probe/samples.json').read_text()):
  if min(len(a['records']) for a in case['arms'])<48:continue
  actions=np.array([[r['delivered_basal_u_h'] for r in a['records']] for a in case['arms']],dtype='float32');history=np.repeat(np.array(case['history'],dtype='float32')[None],5,axis=0);truth=torch.tensor([[r['bg']/18 for r in a['records']] for a in case['arms']],device='cuda');cgm=torch.tensor([[r['cgm']/18 for r in a['records']] for a in case['arms']],device='cuda')
  with torch.no_grad():
   p=agent.patient.rollout(torch.from_numpy(history).cuda(),torch.from_numpy(actions).cuda())
   for h in [12,48]:
    discount=torch.pow(torch.tensor(.9**(1/12),device='cuda'),torch.arange(h,device='cuda'));reward=(p['reward'][:,:h]*discount).sum(1);value=.9**(h/12)*agent.patient.value(p['states'][:,h-1]).squeeze(-1);score=reward+value;glu_reward=(status_score(p['glucose'][:,:h])/12*discount).sum(1);actual=(status_score(truth[:,:h])/12*discount).sum(1);actual_cgm=(status_score(cgm[:,:h])/12*discount).sum(1)
    rows.append({'patient':case['patient'],'minute':case['minute'],'horizon_minutes':h*5,'actions_u_h':actions[:,0].tolist(),'reward_head_return':reward.cpu().tolist(),'discounted_tail_value':value.cpu().tolist(),'reward_plus_tail':score.cpu().tolist(),'glucose_implied_return':glu_reward.cpu().tolist(),'actual_bg_return':actual.cpu().tolist(),'actual_cgm_return':actual_cgm.cpu().tolist(),'chosen_reward':int(reward.argmax()),'chosen_with_tail':int(score.argmax()),'chosen_glucose':int(glu_reward.argmax()),'actual_best':int(actual.argmax()),'actual_regret_reward_only':float(actual.max()-actual[reward.argmax()]),'actual_regret_with_tail':float(actual.max()-actual[score.argmax()])})
 summary={}
 for h in [60,240]:
  rs=[r for r in rows if r['horizon_minutes']==h];summary[h]={'n':len(rs),'reward_only_agreement':float(np.mean([r['chosen_reward']==r['actual_best'] for r in rs])),'with_tail_agreement':float(np.mean([r['chosen_with_tail']==r['actual_best'] for r in rs])),'glucose_implied_agreement':float(np.mean([r['chosen_glucose']==r['actual_best'] for r in rs])),'tail_changes_choice_pct':100*float(np.mean([r['chosen_reward']!=r['chosen_with_tail'] for r in rs])),'reward_regret':float(np.mean([r['actual_regret_reward_only'] for r in rs])),'with_tail_regret':float(np.mean([r['actual_regret_with_tail'] for r in rs]))}
 result={'run':args.run,'weight':args.weight,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'fixed constant-plan development probes; actual comparison covers finite horizon only, not continuation value or human outcomes','summary':summary,'cases':rows};(ROOT/'checks'/(args.run+'_plan_components.json')).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
