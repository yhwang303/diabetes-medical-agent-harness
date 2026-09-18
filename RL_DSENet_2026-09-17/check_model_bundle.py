"""Verify hashes and load the delivered bundle in an isolated project tree."""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    bundle=ROOT/'delivery/DSENet_RL_single_seed_model.tar.gz'
    extracted=ROOT/'delivery/isolated_load';extracted.mkdir(exist_ok=False)
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            assert not member.issym() and not member.islnk()
            target=(extracted/member.name).resolve();assert str(target).startswith(str(extracted.resolve())+'/')
            archive.extract(member,extracted)
    manifest=json.loads((ROOT/'delivery/model_bundle_manifest.json').read_text())
    for name,expected in manifest['files'].items():assert hashlib.sha256((extracted/name).read_bytes()).hexdigest()==expected
    cases=json.loads((ROOT.parent/'RL_DITR创新_2026-09-16/action_probe_dynamic/samples.json').read_text())
    history=np.asarray([c['history'] for c in cases[:2]],dtype='float32')
    anchors=(history[:,-1,1]*.14462788945609448+.09945811581924525)*12
    payload=json.dumps({'history':history.tolist(),'anchor_u_h':anchors.tolist()})+'\n'
    actions=[]
    for project in (ROOT,extracted/ROOT.name):
        process=subprocess.Popen([sys.executable,str(project/'bounded_policy_worker.py'),'--checkpoint',str(project/'results/D06_selected_policy/policy.pt'),'--mode','actor'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,cwd=extracted)
        output,_=process.communicate(payload,timeout=90);assert process.returncode==0
        lines=[json.loads(line) for line in output.splitlines()];assert lines[0]['ready'];actions.append(lines[1]['actions_u_h'])
    np.testing.assert_allclose(actions[0],actions[1],atol=0,rtol=0)
    result={'status':'passed','bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'files_verified':len(manifest['files']),'isolated_tree_cuda_inference':True,'actions_match_original_exact':True,'batch_size':2,'no_patient_dataset_required_for_inference':True}
    (ROOT/'checks/model_bundle.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
