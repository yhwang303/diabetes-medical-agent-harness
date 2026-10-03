import argparse,json,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
ap=argparse.ArgumentParser();ap.add_argument('methods',nargs='+');a=ap.parse_args()
manifest=json.loads((R/'configs/final_manifest.json').read_text());assert not manifest['selection_pending']
for key in a.methods:
 method=next(m for m in manifest['methods'] if m['key']==key)
 for panel in manifest['panels']:
  args=[str(P/'.venv/bin/python'),str(R/'evaluate_final.py'),'--mode',method['mode'],'--config',str(P/manifest['scenario_config']),'--name','C_'+panel+'_'+key,'--batch-size','20']
  if method.get('checkpoint'):
   ck=P/method['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==method['checkpoint_sha256'];args+=['--checkpoint',str(ck)]
  if method['mode']=='actor':args+=['--bounded-reference']
  if method['shared_worker']:args+=['--shared-worker']
  if panel=='brake17':args+=['--brake-config',str(R/'configs/brake17.json')]
  subprocess.run(args,check=True)
print('FINAL_QUEUE_COMPLETE',flush=True)
