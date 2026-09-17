"""Package the frozen patient splits; do not change data or qualification rules."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import pyarrow.parquet as pq

P = Path(__file__).resolve().parents[1]
OUT = P / 'RL数据集'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    with (P / '审计/rl_patient_manifest.csv').open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    expected = json.loads((P / '审计/final_audit.json').read_text())['by_split']
    assert len({r['patient_id'] for r in rows}) == len(rows) == 329
    OUT.mkdir(exist_ok=True)
    results = {}
    for split, counts in expected.items():
        directory = OUT / split
        directory.mkdir(exist_ok=True)
        selected = [r for r in rows if r['split'] == split]
        assert len(selected) == counts['patients']
        transitions = 0
        for row in selected:
            patient = row['patient_id']
            source = P / '患者数据' / (patient + '.parquet')
            dest = directory / source.name
            assert digest(source) == row['file_sha256'], source
            if not dest.exists():
                shutil.copy2(source, dest)
            assert digest(dest) == row['file_sha256'], dest
            table = pq.read_table(dest, columns=['patient_id', 'split', 'rl_transition_eligible'])
            assert table['patient_id'].unique().to_pylist() == [patient], dest
            assert table['split'].unique().to_pylist() == [split], dest
            actual = sum(table['rl_transition_eligible'].to_pylist())
            assert actual == int(row['rl_transitions']), dest
            transitions += actual
        assert {f.stem for f in directory.iterdir()} == {r['patient_id'] for r in selected}
        assert transitions == counts['transitions']
        results[split] = {'patient_files': len(selected), 'eligible_transitions': transitions}
        print(split, results[split], flush=True)
    with (OUT / '患者划分.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        'status': 'verified', 'source': '审计/rl_patient_manifest.csv',
        'manifest_sha256': digest(P / '审计/rl_patient_manifest.csv'),
        'splits': results, 'all_patient_ids_disjoint': True,
        'all_files_sha256_match_source_and_frozen_manifest': True,
        'full_patient_history_preserved': True,
        'all_eligible_transition_counts_rechecked': True,
        'training_target_filter': 'split == train AND rl_transition_eligible == True',
        'scope': 'Packaging only. No new filtering, imputation or model validation.'
    }
    (P / '审计/split_packaging_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
