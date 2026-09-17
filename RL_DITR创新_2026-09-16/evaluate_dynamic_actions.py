"""Equal-total-dose, reordered plans test timing identification beyond constant training arms."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from ditr_data import ROOT
from ditr_model import DITRAgent,status_score

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--weight',default='patient_best.pt');args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=ROOT/'results'/args.run/args.weight;ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 probe=ROOT/'action_probe_dynamic/samples.json';cases=json.loads(probe.read_text());rows=[]
 with torch.no_grad():
  for case in cases:
   arms=case['arms'];a=np.zeros((len(arms),48),dtype='float32');y=np.zeros_like(a);bg=np.zeros_like(a);mask=np.zeros_like(a,dtype=bool)
   for i,arm in enumerate(arms):
    n=len(arm['records']);a[i,:n]=[r['delivered_basal_u_h'] for r in arm['records']];y[i,:n]=[r['cgm']/18 for r in arm['records']];bg[i,:n]=[r['bg']/18 for r in arm['records']];mask[i,:n]=True
   state=np.repeat(np.array(case['history'],dtype='float32')[None],len(arms),axis=0);p=agent.patient.rollout(torch.from_numpy(state).cuda(),torch.from_numpy(a).cuda());g=p['glucose'].cpu().numpy();r=p['reward'].cpu().numpy()
   row={'patient':case['patient'],'minute':case['minute'],'shortened_arms':int((~mask[:,-1]).sum()),'pairs':[]}
   names={arm['pattern']:i for i,arm in enumerate(arms)}
   for first,second in [('early_plus','late_plus'),('early_minus','late_minus'),('plus_then_minus','minus_then_plus')]:
    i,j=names[first],names[second];valid=mask[i]&mask[j];dp=(g[i]-g[j])*18;dt=(y[i]-y[j])*18
    row['pairs'].append({'plans':[first,second],'valid_steps':int(valid.sum()),'path_effect_mae_mg_dl':float(np.abs(dp[valid]-dt[valid]).mean()),'final_true_effect_mg_dl':float(dt[-1]) if valid[-1] else None,'final_predicted_effect_mg_dl':float(dp[-1]) if valid[-1] else None})
   if mask.all():
    discount=np.power(.9**(1/12),np.arange(48));truth=(status_score(torch.tensor(bg)).numpy()/12*discount).sum(1);pred=(r*discount).sum(1);cgmpred=(status_score(torch.tensor(g)).numpy()/12*discount).sum(1)
    chosen=int(pred.argmax());best=int(truth.argmax());row.update({'reward_best':chosen,'actual_best':best,'glucose_implied_best':int(cgmpred.argmax()),'reward_regret':float(truth.max()-truth[chosen]),'glucose_implied_regret':float(truth.max()-truth[cgmpred.argmax()]),'true_status_returns':truth.tolist(),'reward_head_returns':pred.tolist()})
   rows.append(row)
 pairs=[x for row in rows for x in row['pairs']];meaningful=[x for x in pairs if x['final_true_effect_mg_dl'] is not None and abs(x['final_true_effect_mg_dl'])>1];complete=[x for x in rows if 'actual_best' in x]
 summary={'histories':len(rows),'shortened_arms':sum(x['shortened_arms'] for x in rows),'equal_dose_pairs':len(pairs),'path_effect_mae_mg_dl':float(np.mean([x['path_effect_mae_mg_dl'] for x in pairs])),'final_direction_pairs_above_1mg_dl':len(meaningful),'final_direction_agreement':float(np.mean([np.sign(x['final_true_effect_mg_dl'])==np.sign(x['final_predicted_effect_mg_dl']) for x in meaningful])) if meaningful else None,'complete_candidate_sets':len(complete),'reward_choice_agreement':float(np.mean([x['reward_best']==x['actual_best'] for x in complete])) if complete else None,'reward_regret':float(np.mean([x['reward_regret'] for x in complete])) if complete else None,'glucose_implied_regret':float(np.mean([x['glucose_implied_regret'] for x in complete])) if complete else None}
 result={'run':args.run,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'exposed development only; reordered equal-dose plans absent from constant-arm training; no retraining or clinical causal claim','summary':summary,'cases':rows}
 (ROOT/'checks'/(args.run+'_dynamic_actions.json')).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
