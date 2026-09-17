"""Audit downloaded evidence and the final report's local links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
import hashlib,json
root=Path(__file__).resolve().parent
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for key,value in attrs:
   if key in ('href','src'):self.links.append(value)
manifest=json.loads((root/'artifacts/manifest.json').read_text())
assert len(manifest)==8
for a in manifest:
 p=root/'artifacts'/a['file'];assert p.stat().st_size==a['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256']
 assert a['exported_weights_reloaded_exactly_equal']
runs=[]
for run in ['A02_lower_learning_rate','A03_residual_target']:
 p=root/'results'/run;done=json.loads((p/'completion.json').read_text());ev=json.loads((p/'full_validation_evaluation.json').read_text());binding=json.loads((p/'evaluation_binding.json').read_text());diag=json.loads((p/'input_diagnostics.json').read_text());dose=json.loads((p/'local_dose_response.json').read_text())
 assert done['samples_seen']==1653421 and done['full_epochs']==1 and done['patients_seen']==225
 assert ev['per_patient_limit']==0 and ev['horizons_minutes']['5']['model']['n']==424866
 assert ev['horizons_minutes']['30']['model']['n']==166866 and ev['horizons_minutes']['60']['model']['n']==89186
 assert binding['evaluation_sha256']==hashlib.sha256((p/'full_validation_evaluation.json').read_bytes()).hexdigest()
 selected=next(a for a in manifest if a['run']==run and a['kind']=='best')
 assert binding['checkpoint_sha256']==selected['source_full_checkpoint_sha256']==diag['checkpoint_sha256']
 assert done['best_checkpoint_step']==selected['step']==diag['checkpoint_step']==dose['checkpoint_step']
 assert diag['weights_loaded_and_exactly_verified'] and dose['weights_loaded_and_exactly_verified']
 runs.append({'run':run,'checkpoint_step':selected['step'],'all_checks_passed':True})
parser=Links();parser.feed((root/'训练审核报告.html').read_text());broken=[]
for link in parser.links:
 u=urlparse(link)
 if not u.scheme and u.path and not (root/unquote(u.path)).exists():broken.append(link)
assert not broken,broken
result={'artifact_count':len(manifest),'total_weight_bytes':sum(a['bytes'] for a in manifest),'downloaded_sha256_all_match':True,'export_roundtrip_all_exact':True,'runs':runs,'local_report_links_checked':len(parser.links),'broken_links':broken,'clinical_ready':False}
(root/'results/delivery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
