"""Exposed development diagnostics; utility ranking is separate from direction."""
import argparse
import json
import numpy as np
import torch
from world_model import Patient, ROOT, OLD

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='D02_reference_response');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    out=ROOT/'results'/args.name;ck=torch.load(out/'world.pt',map_location='cpu');cfg=ck['config']
    patient=Patient(ROOT/cfg['forecast_checkpoint'],ROOT.parent/cfg['context_checkpoint']).cuda().eval();patient.load_state_dict(ck['model'])
    results={}
    for folder in ('action_probe','action_probe_dynamic'):
        cases=json.loads((OLD/folder/'samples.json').read_text());records=[];all_errors=[];all_true=[];all_pred=[]
        for case in cases:
            history=torch.tensor(np.asarray(case['history'],dtype='float32')[None],device='cuda');z,f=patient.encode(history);rate=patient.reference_rate(history)
            arms=case['arms'];action=torch.zeros(1,len(arms),48,device='cuda');truth=torch.zeros_like(action);mask=torch.zeros_like(action,dtype=torch.bool)
            for i,arm in enumerate(arms):
                n=len(arm['records']);action[0,i,:n]=torch.tensor([r['delivered_basal_u_h'] for r in arm['records']],device='cuda');truth[0,i,:n]=torch.tensor([r['cgm']/18 for r in arm['records']],device='cuda');mask[0,i,:n]=True
            prediction=patient.trajectories(z,f,rate,action);joint=mask[:,1:]&mask[:,0:1]
            delta_true=truth[:,1:]-truth[:,0:1];delta_pred=prediction[:,1:]-prediction[:,0:1]
            all_errors.extend(((delta_pred-delta_true)[joint]*18).cpu().tolist());all_true.extend((delta_true[joint]*18).cpu().tolist());all_pred.extend((delta_pred[joint]*18).cpu().tolist())
            complete=bool(mask.all());record={'patient':case['patient'],'minute':case['minute'],'complete_all_arms':complete,
                  'reference_rmse_mg_dl':float((((prediction[:,0]-truth[:,0])[mask[:,0]])*18).square().mean().sqrt()),
                  'raw_dsenet_reference_rmse_mg_dl':float((((f-truth[:,0])[mask[:,0]])*18).square().mean().sqrt())}
            # Never assign zero reward to missing outcomes; no ranking of truncated arms.
            if complete:
                u=patient.utility(truth)[0];pu=patient.utility(prediction)[0];chosen=int(pu.argmax());oracle=int(u.argmax())
                record.update({'chosen_arm':chosen,'oracle_arm':oracle,'candidate_regret':float(u.max()-u[chosen]),
                    'hold_reference_regret':float(u.max()-u[0]),'utility_mae':float((u-pu).abs().mean()),
                    'chosen_equals_oracle':chosen==oracle,'predicted_utility':pu.cpu().tolist(),'true_utility':u.cpu().tolist()})
            records.append(record)
        ranked=[r for r in records if r['complete_all_arms']];dt=np.array(all_true);dp=np.array(all_pred);meaningful=np.abs(dt)>1
        results[folder]={'histories':len(records),'complete_candidate_pools':len(ranked),
             'effect_path_mae_mg_dl':float(np.mean(np.abs(all_errors))),
             'zero_response_path_mae_mg_dl':float(np.mean(np.abs(dt))),
             'true_effect_rms_mg_dl':float(np.sqrt(np.mean(dt**2))),
             'predicted_effect_rms_mg_dl':float(np.sqrt(np.mean(dp**2))),
             'direction_agreement_above1mgdl':float(np.mean(np.sign(dt[meaningful])==np.sign(dp[meaningful]))),
             'reference_rmse_mg_dl':float(np.mean([r['reference_rmse_mg_dl'] for r in records])),
             'raw_dsenet_reference_rmse_mg_dl':float(np.mean([r['raw_dsenet_reference_rmse_mg_dl'] for r in records])),
             'candidate_regret':float(np.mean([r['candidate_regret'] for r in ranked])) if ranked else None,
             'hold_reference_regret':float(np.mean([r['hold_reference_regret'] for r in ranked])) if ranked else None,
             'oracle_choice_accuracy':float(np.mean([r['chosen_equals_oracle'] for r in ranked])) if ranked else None,
             'records':records}
    gate=all(r['effect_path_mae_mg_dl']<r['zero_response_path_mae_mg_dl'] and r['candidate_regret'] is not None and r['candidate_regret']<r['hold_reference_regret'] for r in results.values())
    result={'scope':'exposed development, not sealed control','decision_gate_passed':gate,
            'gate':'effect path MAE below zero response AND candidate regret below holding reference on both existing probe sets; fixed before reading results',
            'results':results}
    (out/'development_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps({'decision_gate_passed':gate,'results':{k:{a:b for a,b in v.items() if a!='records'} for k,v in results.items()}},indent=2))

if __name__=='__main__':main()
