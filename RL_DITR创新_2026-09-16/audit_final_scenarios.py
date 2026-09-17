"""Audit draft scenario identities before use; does not run a simulator or freeze models."""
import hashlib,json
from pathlib import Path
from evaluate_control import scenario
ROOT=Path(__file__).resolve().parent

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def collect(value,seeds,meal_sequences):
 if isinstance(value,dict):
  for key,item in value.items():
   if key in ['seed','scenario_seed'] and isinstance(item,int):seeds.add(item)
   elif key in ['seeds','scenario_seeds'] and isinstance(item,list):seeds.update(x for x in item if isinstance(x,int))
   elif key=='meals' and isinstance(item,list) and item and isinstance(item[0],list):meal_sequences.append(item)
   elif key not in ['records','source_sha256']:collect(item,seeds,meal_sequences)
 elif isinstance(value,list):
  for item in value:collect(item,seeds,meal_sequences)

def main():
 cfg=json.loads((ROOT/'configs/final_control_draft.json').read_text());old_seeds=set();sequences=[];sources=[]
 for name in ['RL训练_2026-09-15','RL进阶对比_2026-09-15',ROOT.name]:
  base=ROOT.parent/name
  for subdir in ['results','checks','action_probe','action_probe_dynamic','paired_sim_train','paired_sim_train_temporal']:
   for path in sorted((base/subdir).rglob('*.json')):
    # A previous read-only draft audit contains planned seeds but no outcomes.
    # Do not mistake our own recorded proposal for a previously run scenario.
    if path==ROOT/'checks/final_scenario_identity_audit.json':continue
    raw=path.read_bytes();data=json.loads(raw);collect(data,old_seeds,sequences)
    sources.append({'path':str(path.relative_to(ROOT.parent)),'sha256':hashlib.sha256(raw).hexdigest()})
 # Both historical simulation implementations used the same normal-meal generator.
 # Comparing a shared three-day prefix avoids calling a longer copy a new scenario.
 historical_prefixes={digest(scenario(s,4320)) for s in old_seeds}
 historical_prefixes.update(digest([x for x in seq if x[0]<4320]) for seq in sequences)
 cases=[]
 for group in cfg['groups']:
  for seed in group['seeds']:
   meals=scenario(seed,cfg['total_minutes'],group.get('irregular',False));prefix=[x for x in meals if x[0]<4320]
   cases.append({'group':group['name'],'seed':seed,'sensor_seed':seed+10000,'bolus_factor':group['bolus_factor'],'meals':meals,'meals_sha256':digest(meals),'three_day_prefix_sha256':digest(prefix),'seed_seen_before':seed in old_seeds,'three_day_prefix_seen_before':digest(prefix) in historical_prefixes})
 collisions=[x for x in cases if x['seed_seen_before'] or x['three_day_prefix_seen_before']]
 result={'status':'passed' if not collisions else 'collision','scope':'draft known-patient new-scenario identity audit; no outcomes or simulator calls; not a model or protocol freeze','source_files_scanned':len(sources),'historical_named_seeds':sorted(old_seeds),'sources':sources,'config':cfg,'cases':cases,'paired_under_over_same_seed_intentional':True,'all_patients_already_exposed':True,'must_refresh_before_execution':True}
 (ROOT/'checks/final_scenario_identity_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps({'status':result['status'],'files':len(sources),'historical_seed_count':len(old_seeds),'cases':len(cases),'collisions':len(collisions)}))

if __name__=='__main__':main()
