"""Verify legacy weight identity and case-isolated RNG inference, without training."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'RL进阶对比_2026-09-15'
sys.path.insert(0,str(OLD))
from numpy_policy import NumpyPolicy

def main():
    frozen=json.loads((OLD/'configs/T01_frozen_sealed_comparison.json').read_text())
    cases=json.loads((ROOT.parent/'RL_DITR创新_2026-09-16/action_probe_dynamic/samples.json').read_text())
    history=np.asarray([c['history'] for c in cases[:2]],dtype='float32')
    checks={}
    for name,item in frozen['weights'].items():
        path=OLD/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        assert hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()==item['metadata_sha256']
        worker=subprocess.Popen([sys.executable,str(ROOT/'legacy_policy_worker.py'),'--checkpoint',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert json.loads(worker.stdout.readline())['ready']
        reference=[NumpyPolicy(path),NumpyPolicy(path)];rng=[np.random.default_rng(82001+20000),np.random.default_rng(82002+20000)]
        for order in ([0,1],[1,0]):
            payload={'history':history[order].tolist(),'case_keys':[str(i) for i in order],'scenario_seeds':[82001+i for i in order]}
            worker.stdin.write(json.dumps(payload)+'\n');worker.stdin.flush();response=json.loads(worker.stdout.readline())
            expected=[float(np.clip(10*(reference[i](history[i],rng[i].normal(size=(1,1)))[0,0]+1),0,20)) for i in order]
            np.testing.assert_allclose(response['actions_u_h'],expected,atol=0,rtol=0)
        worker.stdin.close();assert worker.wait(timeout=20)==0
        checks[name]={'weights_sha256':item['sha256'],'metadata_sha256':item['metadata_sha256'],'matches_original_numpy_policy_exact':True,'case_reordering_rng_isolated':True}
    (ROOT/'checks/frozen_baseline_workers.json').write_text(json.dumps({'status':'passed','no_training':True,'checks':checks},indent=2));print(json.dumps({'status':'passed','frozen_baselines':len(checks)}))

if __name__=='__main__':main()
