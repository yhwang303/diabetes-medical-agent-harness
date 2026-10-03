"""Freeze only on development results; no confirmation outputs are opened."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
manifest=json.loads((R/'configs/final_manifest_draft.json').read_text())
subprocess.run([str(P/'.venv/bin/python'),str(R/'aggregate.py')],check=True)
dev=json.loads((R/'analysis/development.json').read_text());assert all(k in dev['methods'] for k in ['D_brake17','D_brake30','D_ppo02','D_ppo04','D_ppo08','D_world_tail'])
passing=[k for k,v in dev['joint'].items() if v['passed']]
if passing:raise RuntimeError('Unexpected passing candidate requires explicit existing tie-break mapping before confirmation')
manifest['selected_candidate']='old';manifest['selection_pending']=False;manifest['development_sha256']=hashlib.sha256((R/'analysis/development.json').read_bytes()).hexdigest();manifest['selection_reason']='No candidate passed preregistered strict joint development gate. Retain D06 unchanged.'
for m in manifest['methods']:
 if m['key']=='ours':m['training_information']='Loop + S1 paired simulation and H02 context; retained original, no S2 refinement selected'
 if m['key'] in ['bc','td3bc','iql','rebrac','fql','lom','gfp']:
  out=P/m['checkpoint'];completion=out.parent/'completion.json';deadline=time.time()+7200
  while not completion.exists():
   if (out.parent/'failure.json').exists():raise RuntimeError('Shared training failed '+m['key'])
   if time.time()>deadline:raise TimeoutError(m['key'])
   time.sleep(10)
  done=json.loads(completion.read_text());assert done['steps']==(25000 if m['key']=='lom' else 20000)
 if m['key']=='ditr':assert (P/m['checkpoint']).exists()
 if m['checkpoint']:m['checkpoint_sha256']=hashlib.sha256((P/m['checkpoint']).read_bytes()).hexdigest()
for role,e in json.loads((R/'configs/retained_weights.json').read_text()).items():assert hashlib.sha256((P/'RL_DSENet_2026-09-17'/e['path']).read_bytes()).hexdigest()==e['sha256']
p=R/'configs/final_manifest.json';assert not p.exists();p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('FINAL_SELECTION_FROZEN '+hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
