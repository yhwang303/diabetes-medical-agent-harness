"""Archive the completed frozen set and verify every local archive member.

No experiment execution, extraction, overwrite, or model deserialization.
Development raw data remain in the separately verified development archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

R = Path(__file__).resolve().parent
P = R.parent


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def archive(bundle_path):
    from summarize_confirmation import validate_bundle
    bundle = json.loads(bundle_path.read_text())
    audit = validate_bundle(bundle)
    # Bind current bytes to the already-scored version, not merely to this tar.
    for name, snapshot in audit['metadata'].items():
        assert sha(P / name) == snapshot['sha256'], 'Audited metadata changed: ' + name
    scored_outputs = {}
    for paths in audit['methods'].values():
        finished = json.loads(audit['metadata'][paths['finished']]['text'])
        for name, expected_hash in finished['result_sha256'].items():
            assert sha(P / name) == expected_hash, 'Scored output changed: ' + name
            scored_outputs[name] = expected_hash
    freeze = (P / audit['freeze_path']).resolve()
    freeze.relative_to(R / 'confirmation')
    frozen = json.loads(freeze.read_text())
    queue = R / 'checks' / ('confirmation_queue_' + freeze.parent.name)
    summary = json.loads((queue / 'summary.json').read_text())
    expected = {m['id'] for m in frozen['plan']['methods']}
    assert len(expected) == 18 and summary['all_methods_reported'] is True
    assert {m['method_id'] for m in summary['methods']} == expected
    assert all(m['error'] is None and m['complete_fixed_case_reporting'] for m in summary['methods'])
    files = set()
    bindings = {}

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                bindings[value['path']] = value['sha256']
            for key in ('source_sha256', 'artifact_sha256'):
                if isinstance(value.get(key), dict):
                    for name, expected_hash in value[key].items():
                        if name in bindings:
                            assert bindings[name] == expected_hash
                        bindings[name] = expected_hash
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(frozen['plan'])
    for name, expected_hash in bindings.items():
        path = (P / name).resolve()
        path.relative_to(P)
        assert sha(path) == expected_hash, name
        files.add(path)
    for folder in (freeze.parent, queue):
        for path in folder.rglob('*'):
            if path.is_file():
                assert not path.is_symlink() and path.name != '.gate.lock'
                files.add(path.resolve())
    files.update((bundle_path.resolve(), Path(__file__).resolve(), R / 'execute_confirmation_set.py'))
    target = R / 'archives' / 'confirmation_evidence_r1.tar.gz'
    manifest = R / 'checks' / 'confirmation_evidence_backup_r1.json'
    assert not target.exists() and not manifest.exists()
    entries = {}
    for path in sorted(files):
        relative = str(path.relative_to(P))
        assert path.is_file() and not path.is_symlink()
        entries[relative] = dict(sha256=sha(path), bytes=path.stat().st_size)
    assert set(scored_outputs).issubset(entries)
    assert all(entries[name]['sha256'] == expected_hash for name, expected_hash in scored_outputs.items())
    with target.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz') as tar:
            for name in entries:
                tar.add(P / name, arcname=name, recursive=False)
    for name, entry in entries.items():
        assert sha(P / name) == entry['sha256'], 'Source changed while archiving: ' + name
    result = dict(status='archived_complete_frozen_confirmation', archive=str(target.relative_to(P)),
        archive_sha256=sha(target), archive_bytes=target.stat().st_size, files=entries,
        file_count=len(entries), uncompressed_bytes=sum(x['bytes'] for x in entries.values()),
        frozen_method_count=18, scored_cases=sum(p['episodes'] for p in bundle['panels'].values()),
        frozen_bindings=len(bindings), confirmation_bundle_sha256=sha(bundle_path),
        audited_result_bindings=len(scored_outputs),
        retains_failures_and_unknown_tails=True, development_raw_in_separate_archive=True)
    write_new(manifest, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}))


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    target = P / manifest['archive']
    assert sha(target) == manifest['archive_sha256']
    seen = set()
    with tarfile.open(target, 'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name not in seen
            expected = manifest['files'][member.name]
            assert member.size == expected['bytes']
            with tar.extractfile(member) as stream:
                assert digest(stream) == expected['sha256'], member.name
            seen.add(member.name)
    assert seen == set(manifest['files'])
    result = dict(status='all_archive_members_exact', manifest_sha256=sha(manifest_path),
        archive_sha256=manifest['archive_sha256'], archive_bytes=target.stat().st_size,
        members=len(seen), uncompressed_bytes=manifest['uncompressed_bytes'],
        frozen_method_count=manifest['frozen_method_count'], scored_cases=manifest['scored_cases'],
        extracted=False, retains_failures_and_unknown_tails=True)
    write_new(R / 'checks' / 'confirmation_evidence_local_backup_r1.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('archive', 'verify'))
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    (archive if args.mode == 'archive' else verify)(args.input.resolve())
