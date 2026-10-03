"""Synthetic development metadata only; no real confirmation or model loading."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import confirmation_gate as gate
import evaluate_candidates as evaluator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.relative_to((R/'checks').resolve())
    if output.exists():raise FileExistsError(output)
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))
    sources=[R/'confirmation_gate.py',R/'evaluate_candidates.py',Path(__file__).resolve()]
    source_sha={str(p.relative_to(R)):gate.sha(p) for p in sources}
    ast.parse(sources[0].read_text(),feature_version=(3,8))
    ast.parse(sources[-1].read_text(),feature_version=(3,8))
    check('python38_AST',True)
    protocol=gate.read(R/'protocol.json');jobs=evaluator.make_jobs(protocol,'development')
    with tempfile.TemporaryDirectory(prefix='development_config_fixture_',dir=R/'checks') as temporary:
        project=Path(temporary);research=project/R.name
        write(research/'protocol.json',protocol)
        config=research/'configs/fixture_config.json';write(config,dict(fixture_only=True))
        retained_config=research/'configs/retained_methods.json';write(retained_config,dict(fixture_only=True,role='retained'))
        checkpoint=research/'fixture_checkpoint.bin';checkpoint.write_bytes(b'explicit fixture, not model weights')
        def binding(path):return dict(path=str(path.relative_to(project)),sha256=gate.sha(path))
        with patch.multiple(gate,R=research,P=project):
            for kind in ('hold','physiology','legacy','ppo','retained'):
                folder=research/'results'/('fixture_'+kind);folder.mkdir(parents=True)
                method=dict(id='fixture_'+kind,kind=kind,method='bc' if kind=='retained' else None,
                    checkpoint=None if kind in ('hold','physiology') else binding(checkpoint),
                    config=binding(config) if kind=='ppo' else None,
                    artifact_sha256={str(retained_config.relative_to(project)):gate.sha(retained_config)})
                episodes=[]
                for index,job in enumerate(jobs):
                    path=folder/('raw%02d.json'%index);write(path,dict(fixture_only=True,index=index))
                    episodes.append(dict(key=evaluator.job_key(job),raw_path=path.name,raw_sha256=gate.sha(path)))
                manifest=dict(kind=kind,method=method['method'],split='development',smoke=False,
                    jobs=jobs,jobs_sha256=evaluator.json_sha(jobs),scorer_sha256=protocol['scorer_sha256'],
                    checkpoint_sha256=method['checkpoint']['sha256'] if method['checkpoint'] else None)
                def bind_current():
                    write(folder/'manifest.json',manifest)
                    write(folder/'summary.json',dict(status='completed',count=60,planned_count=60,
                        split='development',smoke=False,jobs_sha256=manifest['jobs_sha256'],episodes=episodes,
                        provenance=dict(manifest_sha256=gate.sha(folder/'manifest.json'))))
                    method['development_evidence']=dict(manifest=binding(folder/'manifest.json'),summary=binding(folder/'summary.json'))
                expected=gate.method_config_sha(method)
                if kind in ('hold','physiology','legacy'):
                    check(kind+'_expected_config_is_None',expected is None)
                    bind_current();gate.development_evidence(method)
                    checks.append(kind+'_missing_config_key_accepted')
                    manifest['config_sha256']=None;bind_current();gate.development_evidence(method)
                    checks.append(kind+'_explicit_null_config_accepted')
                    manifest['config_sha256']='0'*64;bind_current()
                    reject(kind+'_unexpected_config_hash_rejected',lambda:gate.development_evidence(method))
                else:
                    check(kind+'_required_config_hash_nonnull',isinstance(expected,str) and len(expected)==64)
                    bind_current();reject(kind+'_missing_required_config_rejected',lambda:gate.development_evidence(method))
                    manifest['config_sha256']=None;bind_current()
                    reject(kind+'_null_required_config_rejected',lambda:gate.development_evidence(method))
                    manifest['config_sha256']='0'*64;bind_current()
                    reject(kind+'_wrong_required_config_hash_rejected',lambda:gate.development_evidence(method))
                    manifest['config_sha256']=expected;bind_current();gate.development_evidence(method)
                    checks.append(kind+'_exact_required_config_hash_accepted')
    check('sources_unchanged',all(gate.sha(R/name)==digest for name,digest in source_sha.items()))
    check('no_model_or_simulator_imported',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    evidence=dict(status='passed',count=len(checks),checks=checks,source_sha256=source_sha,
        scope='development_evidence config key compatibility only; other metadata remains hash-consistent',
        fixture_only=True,old_manifests_modified=False,actual_selection_frozen=False,
        actual_confirmation_jobs_generated=False,training_simulation_or_remote_run=False)
    gate.write_new(output,evidence)
    print(json.dumps(dict(status='passed',count=len(checks),gate_sha256=source_sha['confirmation_gate.py'])))


if __name__=='__main__':main()
