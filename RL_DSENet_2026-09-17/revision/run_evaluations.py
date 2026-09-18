"""Run a named, frozen method; no tuning, resume only by complete recorded results."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--methods',nargs='+',required=True);args=ap.parse_args()
 c=json.loads((ROOT/'configs/comparison_manifest.json').read_text())
 assert json.loads((ROOT/'checks/evaluator_equivalence.json').read_text())['status']=='passed'
 for m in c['methods']:
  if m['key'] not in args.methods:continue
  if m['mode']=='external':
   ck=PROJECT/m['checkpoint'];done=json.loads((ck.parent/'completion.json').read_text())
   assert done['final_checkpoint_sha256']==hashlib.sha256(ck.read_bytes()).hexdigest()
  for panel in ('main','native'):
   folder=PROJECT/m[panel+'_folder']
   if folder.parent!=ROOT/'results':continue
   if folder.exists():
    assert (folder/'summary.json').exists(),'Partial result needs explicit diagnosis: '+str(folder)
    summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='completed' and summary['count']==60
    continue
   command=[str(PROJECT/'.venv/bin/python'),str(ROOT/'evaluate_extended.py'),'--config',str(ROOT/'configs/scenarios.json'),
            '--checkpoint',str(PROJECT/m['checkpoint']),'--mode',m['mode'],'--name',folder.name,'--batch-size','20']
   if panel=='main':command+=['--common-limit','.25']
   print(json.dumps({'stage':folder.name,'method':m['label'],'panel':panel,'checkpoint_sha256':hashlib.sha256((PROJECT/m['checkpoint']).read_bytes()).hexdigest()}),flush=True)
   subprocess.run(command,check=True)
 print(json.dumps({'status':'queue_complete','methods':args.methods}),flush=True)
if __name__=='__main__':main()
