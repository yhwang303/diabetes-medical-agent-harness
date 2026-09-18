"""Apply the predeclared validation-only criterion after every trial completes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAMES=['D01_forecast','P01_g12s6_l2s1','P02_g24s6_l2s1','P03_g12s3_l2s1','P04_g12s6_l4s1','P05_g12s6_l2s2']

def main():
    rows=[];base=None
    for name in NAMES:
        folder=ROOT/'results'/name
        completion=json.loads((folder/'completion.json').read_text());config=json.loads((folder/'config.json').read_text())
        assert config['seed']==260915 and config['epochs']==3 and config['train_origins']==1653421
        structural={k:v for k,v in config['model'].items() if k not in ('m_patch_len','m_stride','patch_len','stride')}
        if base is None:base=structural
        assert base==structural
        epochs=[json.loads((folder/('validation_epoch%d.json'%e)).read_text()) for e in (1,2,3)]
        best=min(range(3),key=lambda i:epochs[i]['selection_score'])
        full=json.loads((folder/'validation_full.json').read_text())
        rows.append({'name':name,'patch':{k:config['model'][k] for k in ('m_patch_len','m_stride','patch_len','stride')},
                     'selection_score':epochs[best]['selection_score'],'selected_epoch':best+1,
                     'full_validation':full['summary_patient_equal']['dsenet'],'parameters':config.get('parameters',247832),
                     'training_seconds':completion['elapsed_seconds'],'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()})
    selected=min(rows,key=lambda r:(r['selection_score'],r['parameters']))
    result={'selected':selected['name'],'criterion':'minimum four-horizon mean patient-equal RMSE on fixed128 origins per validation patient; exact ties fewer parameters',
            'no_test_used':True,'single_training_seed':260915,'results':rows}
    (ROOT/'checks/patch_selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
