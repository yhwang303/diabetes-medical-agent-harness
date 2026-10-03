"""Freeze tiny completed-file subsets while full sampling jobs are still running."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from prepare_windows import windows

R=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for split,natural_name,paired_name in [
        ('train','natural_train_r1','paired_train_r1'),
        ('world_validation','natural_validation_r1','paired_world_validation_r1')]:
        source=R/'results'/natural_name
        provenance=json.loads((source/'manifest.json').read_text())
        chosen=[j for j in provenance['jobs'] if j['patient']==1 and j['seed']==provenance['jobs'][0]['seed'] and j['bolus_factor'] in (.8,1.2)]
        assert len(chosen)==2
        out=R/'cache'/('smoke_natural_'+split);out.mkdir(exist_ok=False)
        episodes=[]
        for job in chosen:
            key='%s_p%02d_s%d'%(job['group'],job['patient'],job['seed'])
            raw_path=source/(key+'.json');raw=json.loads(raw_path.read_text())
            assert raw['failure_reason'] is None and len(raw['records'])==864
            packed=windows(raw,stride=24)
            path=out/(key+'.npz');np.savez_compressed(path,**packed)
            episodes.append(dict(file=path.name,origins=len(packed['history']),sha256=sha(path),job=job,
                                 source_sha256=sha(raw_path)))
        (out/'manifest.json').write_text(json.dumps(dict(status='completed',split=split,horizon=72,
            kind='natural',episodes=episodes,smoke_only=True,source=natural_name,
            source_manifest_sha256=sha(source/'manifest.json')),indent=2))
        source=R/'cache'/paired_name
        started=json.loads((source/'started.json').read_text())
        chosen=[j for j in started['jobs'] if j['patient']==1 and j['seed']==started['jobs'][0]['seed'] and j['bolus_factor'] in (.8,1.2)]
        assert len(chosen)==2
        out=R/'cache'/('smoke_paired_'+split);out.mkdir(exist_ok=False)
        episodes=[]
        for job in chosen:
            name='p%02d_s%d_b%.1f.npz'%(job['patient'],job['seed'],job['bolus_factor'])
            path=source/name
            with np.load(path,allow_pickle=False) as data:
                groups=len(data['history']);assert data['mask'].shape==(groups,9,72)
                assert groups==8
            shutil.copyfile(path,out/name)
            assert sha(path)==sha(out/name)
            episodes.append(dict(file=name,groups=groups,sha256=sha(path),job=job))
        (out/'manifest.json').write_text(json.dumps(dict(status='completed',kind='paired_interventions',
            split=split,horizon=72,episodes=episodes,smoke_only=True,
            subset_of_running_parent=paired_name,source_started_sha256=sha(source/'started.json')),indent=2))
    print(json.dumps(dict(status='completed',scope='four tiny smoke datasets, not full training datasets')))

if __name__=='__main__':main()
