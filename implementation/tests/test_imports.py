import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from medical_harness.api import create_app
from medical_harness.contracts import canonical
from medical_harness.core import Core

AUTH = {"Authorization": "Bearer alice-token"}

@pytest.fixture
def client(core):
    with TestClient(create_app(core, {"alice-token": "alice", "bob-token": "bob"})) as client:
        yield client


def payload(snapshot, **changes):
    return {"file_name": "case.json", "content": canonical(snapshot), **changes}


def test_import_timeline_restart_and_duplicate(core, client, snapshot):
    data = payload(snapshot)
    result = client.post('/imports', headers=AUTH, json=data)
    assert result.status_code == 201
    case_id = result.json()['case_id']
    repeat = client.post('/imports', headers=AUTH, json=data)
    assert repeat.json() == {**result.json(), 'duplicate': True}
    assert len(client.get('/cases', headers=AUTH).json()) == 1
    timeline = client.get(f'/cases/{case_id}/timeline', headers=AUTH).json()
    assert timeline['snapshot'] == snapshot
    assert timeline['import']['file_hash'] == hashlib.sha256(data['content'].encode()).hexdigest()
    assert timeline['clinical_interpretation'] is False
    assert timeline['content_type'] == 'input_observations'
    assert 'prediction' not in timeline and 'report' not in timeline
    with core.store.tx() as db:
        for table in ('runs', 'jobs', 'artifacts', 'releases'):
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
        event = db.execute("SELECT * FROM events WHERE kind='file_imported'").fetchone()
        assert db.execute('SELECT 1 FROM outbox WHERE event_id=?', (event['id'],)).fetchone()
        assert 'history' not in event['body'] and 'file_name' not in event['body']
    restarted = Core(core.store)
    try:
        assert restarted.input_timeline('alice', case_id) == timeline
    finally:
        restarted.close()


@pytest.mark.parametrize('changed', [
    {'content': '{"source":1,"source":2}'}, {'content': '{"a":NaN}'},
    {'content': '{}'}, {'content': '[]'}, {'content': 'x' * 65537},
    {'file_name': '../case.json'}, {'file_name': 'case.csv'}, {'file_name': 'bad\n.json'},
    {'verified': True},
])
def test_invalid_import_writes_no_case(core, client, snapshot, changed):
    response = client.post('/imports', headers=AUTH, json={**payload(snapshot), **changed})
    assert response.status_code in (413, 422)
    with core.store.tx() as db:
        for table in ('cases', 'snapshots', 'imports'):
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_missing_points_and_units(client, snapshot):
    snapshot['history'][2]['value'] = None
    snapshot['missing_mask'][2] = True
    result = client.post('/imports', headers=AUTH, json=payload(snapshot)).json()
    body = client.get(f"/cases/{result['case_id']}/timeline", headers=AUTH).json()
    assert body['snapshot']['history'][2]['value'] is None
    assert body['snapshot']['missing_mask'][2] is True
    snapshot['unit'] = 'mmol/L'
    assert client.post('/imports', headers=AUTH, json=payload(snapshot)).status_code == 422


def test_input_read_owner_and_browser_boundary(client, snapshot):
    result = client.post('/imports', headers=AUTH, json=payload(snapshot)).json()
    url = f"/cases/{result['case_id']}/timeline"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={'Authorization': 'Bearer bob-token'}).status_code == 404
    assert client.get('/cases', headers={'Authorization': 'Bearer bob-token'}).json() == []
    assert client.get(url, headers={**AUTH, 'Origin': 'null'}).status_code == 403
    assert client.get(url, headers=AUTH).headers['cache-control'] == 'no-store'


def test_import_audit_failure_rolls_back(core, client, snapshot, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError('test audit unavailable')
    monkeypatch.setattr(core, '_event', broken)
    assert client.post('/imports', headers=AUTH, json=payload(snapshot)).status_code == 503
    with core.store.tx() as db:
        for table in ('cases', 'snapshots', 'imports'):
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_concurrent_import_idempotency(core, snapshot):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: core.import_case('alice', payload(snapshot)), range(2)))
    assert results[0]['case_id'] == results[1]['case_id']
    assert sorted(r['duplicate'] for r in results) == [False, True]
    other = core.import_case('bob', payload(snapshot))
    assert other['case_id'] != results[0]['case_id']


def test_updated_snapshot_does_not_reuse_stale_file_provenance(core, snapshot):
    original = payload(snapshot)
    result = core.import_case('alice', original)
    snapshot['history'][0]['value'] = 181.0
    core.update_case('alice', result['case_id'], snapshot)
    assert core.input_timeline('alice', result['case_id'])['import'] is None
    new = core.import_case('alice', original)
    assert new['case_id'] != result['case_id'] and not new['duplicate']

@pytest.mark.parametrize('where', ['file_name', 'content', 'nested_escape'])
def test_import_rejects_unpaired_unicode(core, client, snapshot, where):
    import json
    data = payload(snapshot)
    if where == 'nested_escape':
        snapshot['source_ref'] = '\ud800'
        data['content'] = json.dumps(snapshot)
    else:
        data[where] = '\ud800' + ('.json' if where == 'file_name' else '')
    response = client.post('/imports', headers=AUTH, content=json.dumps(data).encode())
    assert response.status_code == 422
    assert client.get('/cases', headers=AUTH).json() == []
