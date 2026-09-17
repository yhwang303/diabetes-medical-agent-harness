import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from medical_harness.adapters import fixture_prediction
from medical_harness.api import create_app


@pytest.fixture
def client(core):
    with TestClient(create_app(core, {"alice-token": "alice", "bob-token": "bob"})) as value:
        yield value


AUTH = {"Authorization": "Bearer alice-token"}


def create_run(client, snapshot, mode="eval"):
    case = client.post("/cases", headers=AUTH, json=snapshot)
    assert case.status_code == 201
    result = client.post("/runs", headers=AUTH, json={"case_id": case.json()["case_id"],
                        "mode": mode, "idempotency_key": "api-run"})
    assert result.status_code == 201
    return result.json()["id"]


def make_draft(client, run_id):
    job = client.post(f"/runs/{run_id}/draft-jobs", headers=AUTH)
    assert job.status_code == 200
    token = job.json()["capability"]
    result = client.post("/proposal-jobs/" + job.json()["job_id"],
                         headers={"Authorization": "Bearer " + token},
                         json={"sections": ["forecast", "policy", "limitations"]})
    assert result.status_code == 200
    return job.json()


def test_http_chain(client, snapshot):
    run_id = create_run(client, snapshot)
    assert client.post(f"/runs/{run_id}/execute", headers=AUTH).json()["state"] == "SAFETY_ACCEPTED"
    make_draft(client, run_id)
    assert client.post(f"/runs/{run_id}/release", headers=AUTH).status_code == 409
    for role in ["medical", "ethics"]:
        assert client.post(f"/runs/{run_id}/reviews/{role}", headers=AUTH).status_code == 200
    released = client.post(f"/runs/{run_id}/release", headers=AUTH)
    assert released.status_code == 200
    release_id = released.json()["release_id"]
    result = client.get("/releases/" + release_id, headers=AUTH)
    assert result.status_code == 200
    assert result.json()["report"]["report_type"] == "EngineeringFixtureReport"
    assert result.headers["cache-control"] == "no-store"
    assert client.get("/releases/" + release_id, headers={"Authorization": "Bearer bob-token"}).status_code == 404
    assert client.post(f"/runs/{run_id}/cancel", headers=AUTH).status_code == 200
    assert client.get("/releases/" + release_id, headers=AUTH).status_code == 409


def test_auth_origin_and_host(client, snapshot):
    assert client.post("/cases", json=snapshot).status_code == 401
    assert client.post("/cases", headers={"Authorization": "Bearer fake"}, json=snapshot).status_code == 401
    assert client.post("/cases", headers={**AUTH, "Origin": "https://evil.example"}, json=snapshot).status_code == 403
    assert client.post("/cases", headers={**AUTH, "Origin": "null"}, json=snapshot).status_code == 403
    assert client.post("/cases", headers={**AUTH, "Host": "evil.example"}, json=snapshot).status_code == 400


@pytest.mark.parametrize("raw", ['{"case_id":"a","case_id":"b"}', '{"mode":NaN}', '{"verified":true}', '[]'])
def test_invalid_http_json_never_accepted(client, raw):
    response = client.post("/runs", headers=AUTH, content=raw)
    assert response.status_code == 422


def test_size_limit_and_no_input_echo(client):
    response = client.post("/cases", headers=AUTH, content='{"secret":"do-not-echo"}')
    assert response.json() == {"error": "INVALID_SCHEMA"}
    response = client.post("/cases", headers=AUTH, content="x" * 131073)
    assert response.status_code == 413


@pytest.mark.parametrize("path", ["/artifacts", "/mark_prediction_done", "/set_safety_approved", "/finalize", "/rules", "/templates"])
def test_no_privileged_mutation_route(client, path):
    response = client.post(path, headers=AUTH, json={"verified": True, "origin": "model", "force": True})
    assert response.status_code == 404


def test_cannot_self_assign_reviewer_or_release(client, snapshot):
    run_id = create_run(client, snapshot)
    response = client.post(f"/runs/{run_id}/reviews/ethics", headers=AUTH,
                           json={"verdict": "pass", "reviewer": "ethics"})
    assert response.json()["error"] == "CLIENT_VERDICT_FORBIDDEN"
    response = client.post(f"/runs/{run_id}/release", headers=AUTH,
                           json={"verified": True, "report": "应执行"})
    assert response.json()["error"] == "CLIENT_RELEASE_BODY_FORBIDDEN"


def test_proposal_token_cannot_read_cases_or_start_jobs(client, snapshot):
    run_id = create_run(client, snapshot)
    client.post(f"/runs/{run_id}/execute", headers=AUTH)
    job = make_draft(client, run_id)
    scoped = {"Authorization": "Bearer " + job["capability"]}
    assert client.get(f"/runs/{run_id}", headers=scoped).status_code == 401
    assert client.post(f"/runs/{run_id}/release", headers=scoped).status_code == 401
    assert client.post("/proposal-jobs/" + job["job_id"], headers=AUTH,
                       json={"sections": ["forecast", "policy", "limitations"]}).status_code == 403


def test_model_missing_has_no_draft_or_result_leak(client, snapshot):
    run_id = create_run(client, snapshot, mode="research")
    result = client.post(f"/runs/{run_id}/execute", headers=AUTH)
    assert result.status_code == 503
    assert result.json()["error"] == "MODEL_NOT_CONFIGURED"
    assert client.post(f"/runs/{run_id}/draft-jobs", headers=AUTH).status_code == 409
    for suffix in ["", "/events"]:
        status = client.get(f"/runs/{run_id}" + suffix, headers=AUTH)
        assert "action_value" not in status.text
        assert "values" not in status.text
    assert client.get(f"/runs/{run_id}/draft", headers=AUTH).status_code == 404


def test_http_cancel_while_executor_waits(client, core, snapshot):
    run_id = create_run(client, snapshot)
    entered, proceed = threading.Event(), threading.Event()
    def delayed(request):
        entered.set()
        assert proceed.wait(2)
        return fixture_prediction(request)
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=delayed)
    with ThreadPoolExecutor() as pool:
        running = pool.submit(client.post, f"/runs/{run_id}/execute", headers=AUTH)
        assert entered.wait(2)
        try:
            result = client.post(f"/runs/{run_id}/cancel", headers=AUTH)
            assert result.json()["state"] == "CANCELLED"
        finally:
            proceed.set()
        assert running.result().status_code == 409
    assert client.get(f"/runs/{run_id}", headers=AUTH).json()["state"] == "CANCELLED"


def test_cross_case_query_denied(client, snapshot):
    run_id = create_run(client, snapshot)
    result = client.get(f"/runs/{run_id}/events", headers={"Authorization": "Bearer bob-token"})
    assert result.status_code == 404
    events = client.get(f"/runs/{run_id}/events", headers=AUTH).json()
    assert all(event["kind"] != "request_denied" for event in events)
