"""Read-only final forecast evaluation, explicitly separate from train loader."""
import hashlib
import json
import numpy as np
import torch
from forecast_model import Forecast,ROOT
from forecast_data import Data
from train_forecast import evaluate

class FinalData(Data):
    def __init__(self):
        self.split='sealed_test';self.patients={}
        folder=ROOT/'final_forecast_data';manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['final_freeze_sha256']==hashlib.sha256((ROOT/'configs/final_freeze.json').read_bytes()).hexdigest()
        for item in manifest['patients']:
            file=folder/(item['patient_id']+'.npz')
            assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
            with np.load(file) as f:self.patients[item['patient_id']]={k:f[k] for k in f.files}
        self.total=manifest['sample_count']

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text());assert freeze['no_further_selection']
    file=ROOT/freeze['forecast']['path'];assert hashlib.sha256(file.read_bytes()).hexdigest()==freeze['forecast']['sha256']
    out=ROOT/'results/F00_forecast_heldout';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ck=torch.load(file,map_location='cpu');model=Forecast(ck['config']['model']).cuda().eval();model.load_state_dict(ck['model'])
    result=evaluate(model,FinalData(),per_patient=0)
    result['checkpoint_sha256']=freeze['forecast']['sha256'];result['no_model_selection']=True
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary_patient_equal']))

if __name__=='__main__':main()
