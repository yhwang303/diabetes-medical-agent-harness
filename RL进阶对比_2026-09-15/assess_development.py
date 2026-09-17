"""Transparent development checks against the frozen performance tolerances; never final acceptance."""
from pathlib import Path
import json,statistics,hashlib
ROOT=Path(__file__).resolve().parent

def panel(stage,method):
 d=ROOT/'results'/stage
 if not (d/'completion.json').exists():return None
 rows=[r for r in json.loads((d/'summary.json').read_text()) if r['method']==method]
 return {(r['patient'],r['seed'],r['factor']):r for r in rows}

def mean(rows,key):
 patients=sorted({x[0] for x in rows});return statistics.mean(statistics.mean(r[key] for x,r in rows.items() if x[0]==p) for p in patients)

def assess():
 contract=ROOT/'simulation_contract_v3_single_seed.json';rules=json.loads(contract.read_text())['research_acceptance']
 refs={label:panel(stage,method) for label,stage,method in [('nominal','D01_pilot50k_dev','nominal'),('BC','D01_pilot50k_dev','BC'),('recent_RL','D04_rebrac_bc100_dev','ReBRAC_BC100')]}
 candidates=[]
 for folder in sorted((ROOT/'results').glob('D*')):
  if not (folder/'completion.json').exists():continue
  provenance=json.loads((folder/'provenance.json').read_text())
  for method,weight in provenance.get('weights',{}).items():
   cfg=weight.get('config',{});run=cfg.get('name','')
   if (run.startswith('O') or run.startswith('R31_')) and cfg.get('algorithm')!='bc':candidates.append((run+'_step'+str(weight['step']),folder.name,method))

 output=[]
 for label,stage,method in candidates:
  rows=panel(stage,method)
  if not rows:continue
  matched_refs=refs
  if not all(set(rows)==set(r) for r in refs.values()):
   matched_refs={label:panel(stage,method) for label,method in [('nominal','nominal'),('BC','BC'),('recent_RL','ReBRAC_BC100')]}
   assert all(r and set(rows)==set(r) for r in matched_refs.values()),'Unpaired panel'
  checks=[]
  for ref,key in [('BC','delta_TIR_vs_BC_min_pp'),('nominal','delta_TIR_vs_nominal_min_pp'),('recent_RL','delta_TIR_vs_best_recent_RL_min_pp')]:
   value=mean(rows,'tir_pct')-mean(matched_refs[ref],'tir_pct');checks.append({'check':key,'observed':value,'minimum':rules[key],'passed':value>=rules[key]})
  for key,metric in [('TBR70_increase_vs_nominal_and_BC_max_pp','tbr70_pct'),('TBR54_increase_vs_nominal_and_BC_max_pp','tbr54_pct')]:
   value=max(mean(rows,metric)-mean(matched_refs[ref],metric) for ref in ['nominal','BC']);checks.append({'check':key,'observed':value,'maximum':rules[key],'passed':value<=rules[key]})
  failures=sum(r['failed'] for r in rows.values());checks.append({'check':'terminal_failures_allowed','observed':failures,'maximum':rules['terminal_failures_allowed'],'passed':failures<=rules['terminal_failures_allowed']})
  output.append({'candidate':label,'stage':stage,'checks':checks,'preliminary_numeric_screen_passed':all(c['passed'] for c in checks),'final_accepted':False,'interpretation':'historical development screen only; final matched assessment is reported separately', 'pending':([] if stage=='D23_full_development_comparison' else ['full matched development assessment for this candidate'])})
 result={'scope':'partial-development numeric checks only; reference_recent_RL is the current completed strong ReBRAC baseline, not proof of global best','contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'external_baselines_training_seeds':1,'candidate_training_seeds_required':rules['training_seeds_min'],'candidates':output,'clinical_ready':False}
 (ROOT/'results/development_assessment.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(json.dumps(assess(),indent=2))
