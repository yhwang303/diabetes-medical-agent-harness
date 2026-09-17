"""Preserve every attempt's selected/last model weights plus exact training sources and run records."""
from pathlib import Path
import json,hashlib,tarfile
import torch
ROOT=Path(__file__).resolve().parent
out=ROOT/'artifacts';out.mkdir(exist_ok=True);manifest=[]
for run in ['S00_smoke','A01_paper_absolute','A02_lower_learning_rate','A03_residual_target']:
 src=ROOT/'results'/run
 for kind in ['best','last']:
  p=src/(kind+'.pt')
  if not p.exists():continue
  ck=torch.load(p,map_location='cpu');dest=out/(run+'_'+kind+'_model.pt')
  torch.save({'model':ck['model'],'config':ck['config'],'step':ck['step'],'samples_seen':ck.get('samples_seen'),'clinical_ready':False,'release_status':'research_only_not_registered_in_harness','source_full_checkpoint':str(p.relative_to(ROOT))},dest)
  reloaded=torch.load(dest,map_location='cpu')
  assert set(reloaded['model'])==set(ck['model']) and all(torch.equal(v,reloaded['model'][k]) for k,v in ck['model'].items())
  manifest.append({'file':dest.name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'run':run,'kind':kind,'step':ck['step'],'source_full_checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exported_weights_reloaded_exactly_equal':True})
  if kind=='best' and (src/'full_validation_evaluation.json').exists():
   binding={'binding_time':'post_evaluation_export','checkpoint':'best.pt','checkpoint_step':ck['step'],'checkpoint_sha256':manifest[-1]['source_full_checkpoint_sha256'],'evaluation_sha256':hashlib.sha256((src/'full_validation_evaluation.json').read_bytes()).hexdigest(),'note':'Evaluation was run after training ended; best checkpoint was not changed by evaluation. Training provenance remains separate.'}
   (src/'evaluation_binding.json').write_text(json.dumps(binding,indent=2))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
# Metadata snapshot only. All original checkpoints/optimizers remain on the server.
with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as tar:
 for p in (ROOT/'results').rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.log','.py']:tar.add(p,arcname=str(p.relative_to(ROOT)))
 logdir=ROOT.parent/'.autodl-remote/logs'
 for p in logdir.glob('rl-*.log'):tar.add(p,arcname='results/remote_logs/'+p.name)
print(json.dumps(manifest,indent=2))
