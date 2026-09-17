"""Prediction, calibration and same-state intervention diagnostics; no training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from ditr_model import DITRAgent,status_score
from ditr_data import Data,tensor,ROOT

def errors(y,p):
 y=np.asarray(y);p=np.asarray(p);e=p-y
 if not len(e):return {'n':0}
 if not np.isfinite(e).all():return {'n':len(e),'nonfinite':int((~np.isfinite(e)).sum())}
 return {'n':len(e),'mae_mg_dl':float(np.abs(e).mean()*18),'rmse_mg_dl':float(np.sqrt((e*e).mean())*18),'pcc':float(np.corrcoef(y,p)[0,1]) if y.std()>1e-9 and p.std()>1e-9 else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True);ap.add_argument('--per-patient',type=int,default=64);args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 ck=torch.load(args.checkpoint,map_location='cpu');agent=DITRAgent(**ck['config']['model']).cuda().eval();agent.load_state_dict(ck['agent']);agent.requires_grad_(False);data=Data('validation',long_horizon=True);rng=np.random.default_rng(260915);patients=[];horizons=[3,6,12,24,48]
 out=ROOT/'results'/args.name;out.mkdir(parents=True,exist_ok=False)
 with torch.no_grad():
  for pid,d in data.patients.items():
   positions=np.sort(rng.choice(len(d['starts']),min(args.per_patient,len(d['starts'])),replace=False));raw=data.batch(pid,positions,48);b=tensor(raw);pred=agent.patient.rollout(b['state'],b['action']);glu=pred['glucose'].cpu().numpy();probs=pred['wtr'].softmax(-1)[...,1].cpu().numpy();initial=d['glucose'][raw['row']];entry={'patient':pid,'origins':len(positions),'horizons':{}}
   for h in horizons:
    mask=raw['mask'][:,h-1];y=raw['target'][mask,h-1];g=glu[mask,h-1];p=probs[mask,h-1];target=(y*18>=70)&(y*18<=180);common=mask&(raw['length']>=48)
    v={'model':errors(y,g),'persistence':errors(y,initial[mask]),'wtr_brier':float(((p-target)**2).mean()) if len(y) else None,'common_4h_origins_model':errors(raw['target'][common,h-1],glu[common,h-1]),'strata':{}}
    for name,select in [('future_low',raw['target'][:,h-1]*18<70),('future_in_range',(raw['target'][:,h-1]*18>=70)&(raw['target'][:,h-1]*18<=180)),('future_high',raw['target'][:,h-1]*18>180),('food_recorded',(raw['state'][:,:,8]>.5).any(1)),('food_unrecorded',~(raw['state'][:,:,8]>.5).any(1)),('exercise_recorded',(raw['state'][:,:,9]>.5).any(1)),('exercise_unrecorded',~(raw['state'][:,:,9]>.5).any(1))]:
     select=select&mask;v['strata'][name]=errors(raw['target'][select,h-1],glu[select,h-1])
    entry['horizons'][str(h*5)]=v
   patients.append(entry)
  probe_path=ROOT/'action_probe/samples.json';probe=[]
  for case in json.loads(probe_path.read_text()):
   actions=np.zeros((len(case['arms']),48),dtype='float32')
   for i,arm in enumerate(case['arms']):actions[i,:len(arm['records'])]=[x['delivered_basal_u_h'] for x in arm['records']]
   history=np.repeat(np.array(case['history'],dtype='float32')[None],len(actions),axis=0);pred=agent.patient.rollout(torch.from_numpy(history).cuda(),torch.from_numpy(actions).cuda());g=pred['glucose'].cpu().numpy();r=pred['reward'].cpu().numpy();diagnostic={'patient':case['patient'],'minute':case['minute'],'horizons':{},'arms':[],'model_action_conditioning':'actual delivered basal U/h; failed future masked, not synthesized'}
   lengths=[len(a['records']) for a in case['arms']]
   for h in horizons:
    effect_errors=[];bg_errors=[]
    for i in range(1,len(actions)):
     if min(lengths[0],lengths[i])<h:continue
     predicted=(g[i,h-1]-g[0,h-1])*18
     true=case['arms'][i]['records'][h-1]['cgm']-case['arms'][0]['records'][h-1]['cgm'];truebg=case['arms'][i]['records'][h-1]['bg']-case['arms'][0]['records'][h-1]['bg'];effect_errors.append(abs(predicted-true));bg_errors.append(abs(predicted-truebg))
    diagnostic['horizons'][str(h*5)]={'effect_mae_cgm_mg_dl':float(np.mean(effect_errors)) if effect_errors else None,'effect_mae_bg_mg_dl':float(np.mean(bg_errors)) if bg_errors else None,'valid_action_pairs':len(effect_errors)}
   if min(lengths)==48:
    gamma=.9**(1/12);discount=np.power(gamma,np.arange(48));truth=[]
    for arm in case['arms']:
     bg=torch.tensor([r['bg']/18 for r in arm['records']]);truth.append(float((status_score(bg).numpy()/12*discount).sum()))
    predicted=(r*discount).sum(1);chosen=int(np.argmax(predicted));best=int(np.argmax(truth));diagnostic.update({'predicted_best':chosen,'simulator_best':best,'candidate_pool_regret':float(max(truth)-truth[chosen]),'true_status_returns':truth,'predicted_status_returns':predicted.tolist(),'planning_diagnostic_tail_value':False})
   diagnostic['shortened_branches']=sum(n<48 for n in lengths);probe.append(diagnostic)
 summary={}
 for h in horizons:
  key=str(h*5);valid=[p['horizons'][key] for p in patients if p['horizons'][key]['model'].get('rmse_mg_dl') is not None]
  summary[key]={'patients_with_valid_targets':len(valid),'macro_rmse_mg_dl':float(np.mean([p['model']['rmse_mg_dl'] for p in valid])) if valid else None,'macro_mae_mg_dl':float(np.mean([p['model']['mae_mg_dl'] for p in valid])) if valid else None,'persistence_macro_rmse_mg_dl':float(np.mean([p['persistence']['rmse_mg_dl'] for p in valid])) if valid else None,'macro_wtr_brier':float(np.mean([p['wtr_brier'] for p in valid])) if valid else None}
 patient_config=ck['config'];origin=patient_config.get('patient_training_origin')
 if origin:patient_config=json.loads((ROOT/'results'/origin/'config.json').read_text())
 result={'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'checkpoint_stage':ck.get('stage'),'checkpoint_step':ck.get('step'),'patient_training_origin':origin,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_horizon_minutes':patient_config['horizon']*5,'simulation_training_horizon_minutes':patient_config.get('simulation',{}).get('horizon',0)*5,'action_probe_version':'v2_actual_delivered','scope':'logged-future-action-conditioned dynamics; forecasts beyond training horizon explicitly extrapolative; public simulated counterfactuals are not human causal proof','macro_summary':summary,'patients':patients,'action_probes':probe,'interval_calibration':'N/A: point glucose output; WTR probability Brier reported','training_seed':260915,'per_patient_sample':args.per_patient}
 (out/'dynamics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'event':'dynamics_complete','name':args.name,'macro_summary':summary,'probes':len(probe)}),flush=True)

if __name__=='__main__':main()
