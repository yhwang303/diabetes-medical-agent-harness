"""Re-evaluate frozen weights only; final scenarios cannot tune any model."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',choices=['baselines','ditr','ours'],required=True);args=ap.parse_args()
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text());assert freeze['no_further_selection']
    assert json.loads((ROOT/'configs/final_scenarios.json').read_text())==freeze['scenario_config']
    for name,expected in freeze['source_sha256'].items():
        assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==expected,'Source changed after freeze: '+name
    for item in freeze['methods']:
        queue='ditr' if item['mode']=='ditr' else 'ours' if item['bounded_reference'] else 'baselines'
        if queue!=args.queue:continue
        command=[str(ROOT.parent/'.venv/bin/python'),str(ROOT/'evaluate_control.py'),'--config',str(ROOT/'configs/final_scenarios.json'),'--mode',item['mode'],'--name',item['name'],'--batch-size','20']
        if item['path']:
            file=ROOT.parent/item['path'];assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
            command+=['--checkpoint',str(file)]
        if item['bounded_reference']:command+=['--bounded-reference']
        print(json.dumps({'stage':item['name'],'queue':args.queue,'weights_frozen':True}),flush=True)
        subprocess.run(command,check=True)
    print(json.dumps({'status':'final_queue_complete','queue':args.queue}),flush=True)

if __name__=='__main__':main()
