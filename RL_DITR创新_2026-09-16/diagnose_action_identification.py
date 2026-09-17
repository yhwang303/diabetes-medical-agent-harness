"""Measure factual error and intervention-response direction on training/development arms."""
import argparse,json,hashlib
import numpy as np
import torch
from ditr_data import ROOT,tensor
from ditr_model import DITRAgent
from paired_data import PairedData

def group_metrics(pred,true,actions,mask):
 rows=[]
 for h in [6,12,24,48]:
  valid=mask[1:,h-1]&mask[0,h-1];da=actions[1:,h-1]-actions[0,h-1];valid=valid&(np.abs(da)>1e-8)
  dp=(pred[1:,h-1]-pred[0,h-1])*18;dt=(true[1:,h-1]-true[0,h-1])*18
  # Direction is undefined for effectively zero outcomes. Keep their effect errors.
  meaningful=valid&(np.abs(dt)>1.)
  rows.append({'minutes':h*5,'pairs':int(valid.sum()),'effect_sae':float(np.abs(dp[valid]-dt[valid]).sum()),'true_effect_sse':float((dt[valid]**2).sum()),'predicted_effect_sse':float((dp[valid]**2).sum()),'direction_pairs_above_1mg_dl':int(meaningful.sum()),'direction_matches':int((np.sign(dp[meaningful])==np.sign(dt[meaningful])).sum()),'factual_sae':float(np.abs(pred[:,h-1][mask[:,h-1]]-true[:,h-1][mask[:,h-1]]).sum()*18),'factual_n':int(mask[:,h-1].sum())})
 return rows

def summarize(rows):
 result={}
 for h in [30,60,120,240]:
  items=[r for r in rows if r['minutes']==h];total={k:sum(x[k] for x in items) for k in items[0] if k!='minutes'};n=total['pairs'];d=total['direction_pairs_above_1mg_dl']
  result[h]={'pairs':n,'effect_mae_mg_dl':total['effect_sae']/n,'true_effect_rms_mg_dl':(total['true_effect_sse']/n)**.5,'predicted_effect_rms_mg_dl':(total['predicted_effect_sse']/n)**.5,'direction_pairs_above_1mg_dl':d,'direction_agreement':total['direction_matches']/d if d else None,'factual_mae_mg_dl':total['factual_sae']/total['factual_n']}
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--weight',default='patient_best.pt');args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=ROOT/'results'/args.run/args.weight;ck=torch.load(path,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False)
 data=PairedData();rng=np.random.default_rng(260915);indices=np.sort(rng.choice(data.total,min(64,data.total),replace=False));train=[];dev=[]
 with torch.no_grad():
  for offset in range(0,len(indices),4):
   raw=data.batch(indices[offset:offset+4],rng);b=tensor(raw);g=agent.patient.rollout(b['state'],b['action'])['glucose'].cpu().numpy()
   for j in range(0,len(g),5):train+=group_metrics(g[j:j+5],raw['target'][j:j+5],raw['action'][j:j+5],raw['mask'][j:j+5])
  for case in json.loads((ROOT/'action_probe/samples.json').read_text()):
   if min(len(a['records']) for a in case['arms'])<48:continue
   a=np.array([[r['delivered_basal_u_h'] for r in arm['records']] for arm in case['arms']],dtype='float32');y=np.array([[r['cgm']/18 for r in arm['records']] for arm in case['arms']],dtype='float32');state=np.repeat(np.array(case['history'],dtype='float32')[None],5,axis=0)
   g=agent.patient.rollout(torch.from_numpy(state).cuda(),torch.from_numpy(a).cuda())['glucose'].cpu().numpy();dev+=group_metrics(g,y,a,np.ones_like(a,dtype=bool))
 result={'run':args.run,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_groups':indices.tolist(),'training_diagnostic':summarize(train),'development_diagnostic':summarize(dev),'scope':'fixed diagnostics only, not a checkpoint selection rule; direction threshold1mg/dL excludes near-zero directions but retains all effect errors'}
 (ROOT/'checks'/(args.run+'_action_identification.json')).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
