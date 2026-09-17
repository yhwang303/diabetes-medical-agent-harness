"""Regressions for the pre-desktop TODO audit; use only public test inputs."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from medical_harness.adapters import fixture_policy
from medical_harness.api import create_app
from medical_harness.contracts import ErrorResponse, GateError, SnapshotInput, canonical, contract_catalog, digest
from medical_harness.paths import ROOT
from medical_harness.store import Store


@pytest.mark.parametrize("reason", ["model_error", "unsupported", "insufficient_input"])
def test_candidate_cannot_claim_failure(core, run, reason):
    core._fixtures["policy"] = replace(core._fixtures["policy"], call=lambda request: {
        **fixture_policy(request), "reason": reason})
    with pytest.raises(GateError, match="INVALID_MODEL_OUTPUT"):
        core.execute_models("alice", run["id"])
    with core.store.tx() as db:
        assert db.execute("SELECT COUNT(*) FROM artifacts WHERE kind='policy'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM releases").fetchone()[0] == 0
    with pytest.raises(GateError):
        core.create_draft_job("alice", run["id"])


@pytest.mark.parametrize("path", ["/cases", "/proposal-jobs/unknown"])
def test_non_ascii_credential_is_fixed_rejection(core, path):
    with TestClient(create_app(core, {"alice-token": "alice"}), raise_server_exceptions=False) as client:
        response = client.post(path, headers=[(b"authorization", b"Bearer \xe9")], content=b"{}")
        assert response.status_code == 401
        assert response.json() == {"error": "AUTH_REQUIRED"}


def test_schema_rejection_is_audited_without_input(core):
    with TestClient(create_app(core, {"alice-token": "alice"})) as client:
        response = client.post("/cases", headers={"Authorization": "Bearer alice-token"},
                               json={"source_ref": "private-test-do-not-echo"})
        assert response.status_code == 422
        ErrorResponse.model_validate(response.json())
    with core.store.tx() as db:
        event = db.execute("SELECT * FROM events WHERE kind='request_denied'").fetchone()
        assert json.loads(event["body"]) == {"code": "INVALID_SCHEMA"}
        assert db.execute("SELECT 1 FROM outbox WHERE event_id=?", (event["id"],)).fetchone()


def test_schema_rejection_audit_failure_is_not_silenced(core, monkeypatch):
    import sqlite3
    def offline(*args):
        raise sqlite3.OperationalError("private detail")
    monkeypatch.setattr(core, "record_denial", offline)
    with TestClient(create_app(core, {"alice-token": "alice"})) as client:
        response = client.post("/cases", headers={"Authorization": "Bearer alice-token"}, json={})
        assert response.status_code == 503
        assert response.json() == {"error": "AUDIT_UNAVAILABLE"}


@pytest.mark.parametrize("headers,path,method,status,code", [
    ({"Origin": ""}, "/health", "GET", 403, "BROWSER_ORIGIN_FORBIDDEN"),
    ({"Host": "evil.example"}, "/health", "GET", 400, "INVALID_HOST"),
    ({}, "/unknown", "GET", 404, "NOT_FOUND"),
    ({}, "/health", "POST", 405, "METHOD_NOT_ALLOWED"),
])
def test_transport_errors_follow_fixed_contract(core, headers, path, method, status, code):
    with TestClient(create_app(core, {})) as client:
        response = client.request(method, path, headers=headers)
        assert response.status_code == status
        assert response.json() == {"error": code}
        ErrorResponse.model_validate(response.json())
        if status == 405:
            assert "GET" in response.headers["allow"]


@pytest.mark.parametrize("action", ["execute", "cancel", "draft-jobs"])
def test_no_input_actions_reject_client_facts(core, run, action):
    with TestClient(create_app(core, {"alice-token": "alice"})) as client:
        response = client.post(f"/runs/{run['id']}/{action}", headers={"Authorization": "Bearer alice-token"},
                               json={"verified": True, "origin": "model", "action_value": 1})
        assert response.status_code == 422
        assert response.json() == {"error": "CLIENT_BODY_FORBIDDEN"}
    assert core.status("alice", run["id"])["state"] == "SNAPSHOT_FROZEN"
    with core.store.tx() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_simulation_data_cannot_be_relabelled_as_fixture_by_mode(core, snapshot):
    snapshot["source"] = "simulation"
    assert SnapshotInput.model_validate(snapshot).source == "simulation"
    case = core.create_case("alice", snapshot)
    with pytest.raises(GateError, match="FIXTURE_REQUIRES_SYNTHETIC_CASE"):
        core.start_run("alice", case["case_id"], "eval", "simulation")
    snapshot["origin"] = "model"
    with pytest.raises(ValidationError):
        core.create_case("alice", snapshot)


@pytest.mark.parametrize("origin", ["model", "simulator", "unknown"])
def test_executor_origin_must_match_run_before_call(core, run, origin):
    def forbidden(request):
        pytest.fail("mismatched origin executor must not be invoked")
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], origin=origin, call=forbidden)
    with pytest.raises(GateError, match="EXECUTOR_ORIGIN_MISMATCH"):
        core.execute_models("alice", run["id"])


@pytest.mark.parametrize("field,value", [("job_id", "wrong-job"), ("attempt", 2),
                                        ("input_digest", "wrong-input")])
def test_ledger_metadata_corruption_is_detected(core, run, field, value):
    # Storage-integrity check, not a claim to resist an administrator controlling SQLite.
    core.execute_models("alice", run["id"])
    with core.store.tx() as db:
        row = db.execute("SELECT * FROM artifacts WHERE kind='prediction'").fetchone()
        body = json.loads(row["body"])
        body[field] = value
        db.execute("UPDATE artifacts SET body=?,digest=? WHERE id=?", (canonical(body), digest(body), row["id"]))
    with pytest.raises(GateError, match="ARTIFACT_JOB_BINDING"):
        core.create_draft_job("alice", run["id"])


def test_contract_export_matches_runtime(core):
    saved = json.loads((ROOT / "contracts/harness-engineering-v1.json").read_text())
    assert saved == contract_catalog()
    with TestClient(create_app(core, {})) as client:
        assert client.get("/contracts").json() == saved


def test_database_cannot_write_outside_project(tmp_path):
    outside = ROOT.parent / "audit-outside-must-not-exist.sqlite3"
    assert not outside.exists()
    with pytest.raises(ValueError, match="inside implementation"):
        Store(outside)
    link = tmp_path / "escape"
    link.symlink_to(ROOT.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="inside implementation"):
        Store(link / outside.name)
    assert not outside.exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_database_sidecar_symlink_cannot_escape(tmp_path, suffix):
    database = tmp_path / "harness.sqlite3"
    target = ROOT.parent / "audit-sidecar-must-not-exist"
    assert not target.exists()
    Path(str(database) + suffix).symlink_to(target)
    with pytest.raises(ValueError, match="inside implementation"):
        Store(database)
    assert not target.exists()
