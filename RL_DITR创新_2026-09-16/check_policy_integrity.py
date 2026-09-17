"""Verify full-data policy-only training without changing any checkpoint."""
import argparse,hashlib,json
import torch
from ditr_data import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--code',required=True);args=ap.parse_args();cfg=json.loads((ROOT/'configs'/(args.code+'.json')).read_text());folder=ROOT/'results'/cfg['name'];path=folder/'policy_last.pt';ck=torch.load(path,map_location='cpu');initial=torch.load(ROOT/cfg['initialize_checkpoint'],map_location='cpu');completion=json.loads((folder/'policy_completion.json').read_text());assert completion['full_epochs']==1 and completion['samples_seen']==1653421 and completion['patients_seen']==225;assert ck['step']==6580
 old=initial['agent'];new=ck['agent'];patient_keys=[k for k in old if k.startswith('patient.')];assert all(torch.equal(old[k],new[k]) for k in patient_keys);assert all(torch.isfinite(v).all() for v in new.values());change=sum(float((new[k]-old[k]).square().sum()) for k in old if k.startswith('policy.'))**.5;assert change>0
 result={'status':'passed','code':args.code,'training_seed':cfg['seed'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'initial_checkpoint':cfg['initialize_checkpoint'],'initial_sha256':hashlib.sha256((ROOT/cfg['initialize_checkpoint']).read_bytes()).hexdigest(),'patient_parameters_and_buffers_exact':True,'patient_state_tensors_checked':len(patient_keys),'all_final_tensors_finite':True,'policy_parameter_l2_change':change,'completion':completion};(ROOT/'checks'/(args.code+'_training_integrity.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
