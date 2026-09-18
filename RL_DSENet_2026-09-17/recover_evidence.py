"""Read-only inventory and non-overwriting archive of yesterday's evidence."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'RL_DITR创新_2026-09-16'

def main():
    output = ROOT / 'recovery'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'previous_evidence.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    files = set()
    for folder in ('results', 'checks', 'configs'):
        files.update((OLD / folder).rglob('*.json'))
        files.update((OLD / folder).rglob('*.jsonl'))
    selected = {
        'H05_response_delayed_policy': 'policy_last.pt',
        'R08_H02_delayed_policy': 'policy_004000.pt',
        'C02_H02_point_reward': 'patient_composed.pt',
        'P05_temporal_conditioned_response': 'response_last.pt',
        'P06_temporal_direct_prefix': 'response_last.pt',
    }
    inventory = {}
    for run, checkpoint in selected.items():
        folder = OLD / 'results' / run
        path = folder / checkpoint
        if path.exists():
            files.add(path)
        history = folder / 'history.jsonl'
        inventory[run] = {
            'files': sorted(p.name for p in folder.iterdir()),
            'last_history': json.loads(history.read_text().splitlines()[-1]) if history.exists() else None,
            'has_policy_completion': (folder / 'policy_completion.json').exists(),
        }
    manifest = [{'path': str(p.relative_to(OLD)), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(OLD)))
    record = {'inventory': inventory, 'files': manifest,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_bytes': archive.stat().st_size,
              'status': 'evidence_recovered_no_training_started'}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'inventory': inventory, 'file_count': len(files),
                      'archive_bytes': record['archive_bytes']}, indent=2))

if __name__ == '__main__':
    main()
