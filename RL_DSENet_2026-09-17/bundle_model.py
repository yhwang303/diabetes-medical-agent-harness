"""Build a project-relative research inference bundle with its full dependency chain."""
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    freeze=json.loads((ROOT/'configs/final_freeze.json').read_text())
    files=[ROOT/freeze[k]['path'] for k in ('forecast','world','policy')]
    world_config=json.loads((files[1].parent/'config.json').read_text())
    files += [ROOT.parent/world_config['context_checkpoint'],ROOT.parent/'Loop数据集/训练管线_v2/prepared/normalization.json']
    files += [ROOT/name for name in ('forecast_model.py','world_model.py','policy_bounded.py','bounded_policy_worker.py','README.md','实验协议.md','configs/final_freeze.json','checks/environment_and_sources.json')]
    files += [ROOT.parent/'RL_DITR创新_2026-09-16'/name for name in ('ditr_model.py','response_operator.py')]
    files += [p for p in (ROOT/'dsenet').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'results'/name/'config.json' for name in ('P03_g12s3_l2s1','D05_selected_world','D06_selected_policy')]
    files=sorted(set(files))
    manifest={'purpose':'research-only standalone project-relative CUDA inference bundle; no dataset or credentials','files':{str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'frozen_inference_weights':{k:freeze[k] for k in ('forecast','world','policy')}}
    output=ROOT/'delivery';output.mkdir(exist_ok=True)
    (output/'model_bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(output/'DSENet_RL_single_seed_model.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT.parent)))
        archive.add(output/'model_bundle_manifest.json',arcname='RL_DSENet_2026-09-17/delivery/model_bundle_manifest.json')
    print(json.dumps({'files':len(files),'bytes':(output/'DSENet_RL_single_seed_model.tar.gz').stat().st_size,'sha256':hashlib.sha256((output/'DSENet_RL_single_seed_model.tar.gz').read_bytes()).hexdigest()}))

if __name__=='__main__':main()
