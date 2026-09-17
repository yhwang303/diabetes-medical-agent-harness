import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pydantic import ValidationError

from medical_harness.adapters import fixture_policy, fixture_prediction, fixture_review
from medical_harness.contracts import GateError, PolicyOutput, SnapshotInput, canonical, strict_json
from medical_harness.core import Core
from medical_harness.render import KB_VERSION, TEMPLATE_VERSION


def draft(core, run_id, order=None):
    job = core.create_draft_job("alice", run_id)
    result = core.submit_proposal(job["job_id"], job["capability"], canonical({
        "sections": order or ["forecast", "policy", "limitations"]}))
    return result


def reviewed(core, run_id):
    core.execute_models("alice", run_id)
    result = draft(core, run_id)
    for role in ["medical", "ethics"]:
        core.review("alice", run_id, role)
    return result


def count(core, table, run_id=None):
    with core.store.tx() as db:
        if run_id:
            return db.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (run_id,)).fetchone()[0]
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_full_chain_and_restart(core, run):
    reviewed(core, run["id"])
    release = core.release("alice", run["id"])
    body = core.get_release("alice", release["release_id"])
    assert body["report"]["origin"] == "fixture"
    assert body["report"]["clinical_use"] is False
    assert len(body["review_ids"]) == 2
    assert core.release("alice", run["id"]) == release
    assert count(core, "releases") == 1
    restarted = Core(core.store, enable_fixtures=True)
    try:
        assert restarted.recover_interrupted() == 0
        assert restarted.get_release("alice", release["release_id"]) == body
    finally:
        restarted.close()


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"x":{"a":1,"a":2}}', '{broken'])
def test_strict_json(raw):
    with pytest.raises(GateError, match="INVALID_JSON"):
        strict_json(raw)


@pytest.mark.parametrize("change", [
    lambda x: x.update(unit="mmol/L"),
    lambda x: x.update(sampling_minutes="5"),
    lambda x: x.update(verified=True),
    lambda x: x.update(decision_time="2026-09-09T10:00:00"),
    lambda x: x["history"][-1].update(value=float("nan")),
    lambda x: x["history"][-1].update(value=True),
    lambda x: x["history"][-1].update(time="2026-09-09T10:05:00+08:00"),
    lambda x: x["history"][1].update(time=x["history"][0]["time"]),
    lambda x: x["missing_mask"].__setitem__(0, True),
])
def test_invalid_snapshot(snapshot, change):
    change(snapshot)
    with pytest.raises(ValidationError):
        SnapshotInput.model_validate(snapshot)


def test_missing_data_never_calls_models(core, snapshot):
    snapshot["history"][-1]["value"] = None
    snapshot["missing_mask"][-1] = True
    case = core.create_case("alice", snapshot)
    run = core.start_run("alice", case["case_id"], "eval", "missing")
    assert core.execute_models("alice", run["id"])["reason"] == "MISSING_INPUT"
    assert count(core, "jobs") == count(core, "artifacts") == 0


def test_research_never_falls_back(core, snapshot):
    case = core.create_case("alice", snapshot)
    run = core.start_run("alice", case["case_id"], "research", "research")
    with pytest.raises(GateError, match="MODEL_NOT_CONFIGURED"):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 0
    assert core.status("alice", run["id"])["state"] == "UNAVAILABLE"


def test_fixtures_opt_in_and_synthetic_only(core, snapshot):
    case = core.create_case("alice", snapshot)
    core.enable_fixtures = False
    with pytest.raises(GateError, match="FIXTURES_DISABLED"):
        core.start_run("alice", case["case_id"], "eval", "disabled")
    core.enable_fixtures = True
    snapshot["source"] = "historical"
    case = core.create_case("alice", snapshot)
    with pytest.raises(GateError, match="FIXTURE_REQUIRES_SYNTHETIC_CASE"):
        core.start_run("alice", case["case_id"], "eval", "historical")


def test_idempotency_conflict_and_owner(core, run):
    assert core.start_run("alice", run["case_id"], "eval", "run-1")["id"] == run["id"]
    with pytest.raises(GateError, match="IDEMPOTENCY_CONFLICT"):
        core.start_run("alice", run["case_id"], "research", "run-1")
    for call in [lambda: core.status("bob", run["id"]), lambda: core.cancel("bob", run["id"]),
                 lambda: core.events("bob", run["id"]), lambda: core.execute_models("bob", run["id"])]:
        with pytest.raises(GateError, match="NOT_FOUND"):
            call()


def test_consumers_require_predecessors(core, run):
    with pytest.raises(GateError):
        core._model_step("alice", run["id"], "policy")
    with pytest.raises(GateError):
        core.create_draft_job("alice", run["id"])
    with pytest.raises(GateError):
        core.review("alice", run["id"], "medical")
    with pytest.raises(GateError):
        core.release("alice", run["id"])
    assert count(core, "jobs") == 0


@pytest.mark.parametrize("mutation,expected", [
    (lambda x: x.update(input_digest="forged"), "INPUT_DIGEST_MISMATCH"),
    (lambda x: x.update(verified=True), "INVALID_MODEL_OUTPUT"),
    (lambda x: x.update(values=[]), "INVALID_MODEL_OUTPUT"),
    (lambda x: x.update(status="error"), "INVALID_MODEL_OUTPUT"),
    (lambda x: x.update(values=[float("inf")] * 6), "INVALID_MODEL_OUTPUT"),
])
def test_invalid_prediction_blocks_policy(core, run, mutation, expected):
    def bad(request):
        output = fixture_prediction(request)
        mutation(output)
        return output
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=bad)
    with pytest.raises(GateError, match=expected):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 0
    with core.store.tx() as db:
        assert not db.execute("SELECT 1 FROM jobs WHERE step='policy'").fetchone()


def test_policy_parent_mismatch(core, run):
    core._fixtures["policy"] = replace(core._fixtures["policy"], call=lambda r: {
        **fixture_policy(r), "forecast_parent_hash": "another-run-prediction"})
    with pytest.raises(GateError, match="FORECAST_PARENT_MISMATCH"):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 1


@pytest.mark.parametrize("status", ["abstain", "unsupported", "error"])
def test_abstention_has_no_action_or_release(core, run, status):
    core._fixtures["policy"] = replace(core._fixtures["policy"], call=lambda r: {
        **fixture_policy(r), "status": status, "action_value": None,
        "reason": {"abstain": "insufficient_input", "unsupported": "unsupported", "error": "model_error"}[status]})
    with pytest.raises(GateError, match="POLICY_" + status.upper()):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 1
    with pytest.raises(GateError):
        core.create_draft_job("alice", run["id"])
    assert count(core, "releases") == 0


def test_zero_is_valid_candidate_not_failure(core, run):
    core._fixtures["policy"] = replace(core._fixtures["policy"], call=lambda r: {
        **fixture_policy(r), "action_value": 0.0})
    assert core.execute_models("alice", run["id"])["state"] == "SAFETY_ACCEPTED"
    invalid = {**fixture_policy({"input_digest": "x", "prediction_hash": "y"}), "status": "abstain", "action_value": 0.0}
    with pytest.raises(ValidationError):
        PolicyOutput.model_validate(invalid)


def test_engineering_safety_rejects_action_without_executing(core, run):
    core._fixtures["policy"] = replace(core._fixtures["policy"], call=lambda r: {
        **fixture_policy(r), "action_value": 0.5})
    assert core.execute_models("alice", run["id"])["reason"] == "ENGINEERING_SAFETY_REJECTED"
    assert "action_value" not in canonical(core.status("alice", run["id"]))
    assert "action_value" not in canonical(core.events("alice", run["id"]))
    with pytest.raises(GateError):
        core.create_draft_job("alice", run["id"])


def test_timeout_retry_and_bounded_attempts(core, run):
    def timeout(_):
        raise GateError("EXECUTOR_TIMEOUT", 503)
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=timeout)
    for _ in range(3):
        with pytest.raises(GateError, match="EXECUTOR_TIMEOUT"):
            core.execute_models("alice", run["id"])
    with pytest.raises(GateError, match="ATTEMPT_BUDGET_EXHAUSTED"):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 0


def test_actual_timeout_discards_late_result(core, run):
    core.executor_timeout = 0.02
    def slow(request):
        time.sleep(0.1)
        return fixture_prediction(request)
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=slow)
    with pytest.raises(GateError, match="EXECUTOR_TIMEOUT"):
        core.execute_models("alice", run["id"])
    core.executor_timeout = 1
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=fixture_prediction)
    assert core.execute_models("alice", run["id"])["state"] == "SAFETY_ACCEPTED"
    time.sleep(0.12)
    assert count(core, "artifacts") == 2
    with core.store.tx() as db:
        attempts = db.execute("SELECT attempt,status FROM jobs WHERE step='prediction' ORDER BY attempt").fetchall()
        assert [tuple(x) for x in attempts] == [(1, "FAILED"), (2, "ACCEPTED")]


def test_cancel_during_model_cannot_resurrect(core, run):
    entered, proceed = threading.Event(), threading.Event()
    def delayed(request):
        entered.set()
        assert proceed.wait(2)
        return fixture_prediction(request)
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=delayed)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(core.execute_models, "alice", run["id"])
        assert entered.wait(2)
        core.cancel("alice", run["id"])
        proceed.set()
        with pytest.raises(GateError, match="CANCELLED"):
            future.result()
    assert count(core, "artifacts") == 0
    assert core.status("alice", run["id"])["state"] == "CANCELLED"


def test_only_one_concurrent_model_job(core, run):
    entered, proceed = threading.Event(), threading.Event()
    def delayed(request):
        entered.set()
        assert proceed.wait(2)
        return fixture_prediction(request)
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], call=delayed)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(core.execute_models, "alice", run["id"])
        assert entered.wait(2)
        try:
            with pytest.raises(GateError, match="INVALID_STAGE|JOB_IN_PROGRESS"):
                core.execute_models("alice", run["id"])
        finally:
            proceed.set()
        future.result()
    assert count(core, "artifacts") == 2


def test_proposal_capabilities_and_no_free_prose(core, run):
    core.execute_models("alice", run["id"])
    job = core.create_draft_job("alice", run["id"])
    good = canonical({"sections": ["forecast", "policy", "limitations"]})
    with pytest.raises(GateError, match="INVALID_CAPABILITY"):
        core.submit_proposal(job["job_id"], "invented", good)
    for field in ["text", "dose", "verified", "html", "template_version"]:
        with pytest.raises(ValidationError):
            core.submit_proposal(job["job_id"], job["capability"], canonical({
                "sections": ["forecast", "policy", "limitations"], field: "现在每天执行"}))
    core.submit_proposal(job["job_id"], job["capability"], good)
    with pytest.raises(GateError, match="STALE_ATTEMPT"):
        core.submit_proposal(job["job_id"], job["capability"], good)


def test_missing_reviewer_and_revision_invalidates_pass(core, run):
    core.execute_models("alice", run["id"])
    draft(core, run["id"])
    core.review("alice", run["id"], "medical")
    with pytest.raises(GateError, match="REVIEW_REQUIRED"):
        core.release("alice", run["id"])
    core.review("alice", run["id"], "ethics")
    draft(core, run["id"], ["limitations", "forecast", "policy"])
    with pytest.raises(GateError, match="REVIEW_REQUIRED"):
        core.release("alice", run["id"])


@pytest.mark.parametrize("verdict", ["fail", "abstain"])
def test_review_rejection_never_majority_votes(core, run, verdict):
    core.execute_models("alice", run["id"])
    draft(core, run["id"])
    core._fixtures["ethics"] = replace(core._fixtures["ethics"], call=lambda r: {
        **fixture_review(r), "verdict": verdict, "issues": ["fixture_review_failure"]})
    core.review("alice", run["id"], "medical")
    assert core.review("alice", run["id"], "ethics")["state"] == "REVIEW_BLOCKED"
    with pytest.raises(GateError):
        core.release("alice", run["id"])


def test_review_hash_forgery(core, run):
    core.execute_models("alice", run["id"])
    draft(core, run["id"])
    core._fixtures["medical"] = replace(core._fixtures["medical"], call=lambda r: {
        **fixture_review(r), "report_hash": "old-report"})
    with pytest.raises(GateError, match="REVIEW_BINDING"):
        core.review("alice", run["id"], "medical")
    assert count(core, "reviews") == 0


def test_revision_budget(core, run):
    core.execute_models("alice", run["id"])
    for _ in range(3):
        draft(core, run["id"])
    with pytest.raises(GateError, match="REVISION_BUDGET_EXHAUSTED"):
        core.create_draft_job("alice", run["id"])


@pytest.mark.parametrize("invalidate", ["snapshot", "template", "knowledge", "expiry", "cancel"])
def test_read_and_publish_recheck_current_validity(core, run, snapshot, invalidate):
    reviewed(core, run["id"])
    release = core.release("alice", run["id"])
    if invalidate == "snapshot":
        snapshot["history"][-1]["value"] = 126.0
        core.update_case("alice", run["case_id"], snapshot)
    elif invalidate in {"template", "knowledge"}:
        core.revoke_version(TEMPLATE_VERSION if invalidate == "template" else KB_VERSION)
    elif invalidate == "expiry":
        core.clock = lambda: time.time() + 7200
    else:
        core.cancel("alice", run["id"])
    with pytest.raises(GateError):
        core.get_release("alice", release["release_id"])
    with pytest.raises(GateError):
        core.release("alice", run["id"])
    assert core.status("alice", run["id"])["currently_valid"] is False


def test_post_review_byte_tampering(core, run):
    reviewed(core, run["id"])
    with core.store.tx() as db:
        db.execute("UPDATE drafts SET body=replace(body,'未执行','应执行') WHERE run_id=?", (run["id"],))
    with pytest.raises(GateError, match="DRAFT_INTEGRITY"):
        core.release("alice", run["id"])
    assert count(core, "releases") == 0


def test_release_audit_failure_rolls_back(core, run, monkeypatch):
    reviewed(core, run["id"])
    original = core._event
    def fail(db, run_id, kind, **metadata):
        if kind == "release_committed":
            raise sqlite3.OperationalError("disk full")
        return original(db, run_id, kind, **metadata)
    monkeypatch.setattr(core, "_event", fail)
    with pytest.raises(sqlite3.OperationalError):
        core.release("alice", run["id"])
    assert count(core, "releases") == 0
    assert core.status("alice", run["id"])["state"] == "REVIEWED"


def test_concurrent_release_is_single_commit(core, run):
    reviewed(core, run["id"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: core.release("alice", run["id"]), range(4)))
    assert len({r["release_id"] for r in results}) == 1
    assert count(core, "releases") == 1
    assert sum(e["kind"] == "release_committed" for e in core.events("alice", run["id"])) == 1


def test_outbox_offline_and_replay(core, run):
    reviewed(core, run["id"])
    release = core.release("alice", run["id"])
    before = count(core, "events")
    def disconnected(_):
        raise ConnectionError("offline")
    with pytest.raises(ConnectionError):
        core.deliver_outbox(disconnected)
    assert core.get_release("alice", release["release_id"])
    received = {}
    assert core.deliver_outbox(lambda e: received.setdefault(e["id"], e)) == before
    assert len(received) == before
    assert core.deliver_outbox(lambda _: pytest.fail("duplicate delivered after ack")) == 0
    assert "action_value" not in canonical(received)


def test_recovery_fences_unfinished_proposal(core, run):
    core.execute_models("alice", run["id"])
    job = core.create_draft_job("alice", run["id"])
    assert core.recover_interrupted() == 1
    with pytest.raises(GateError, match="STALE_ATTEMPT"):
        core.submit_proposal(job["job_id"], job["capability"], canonical({"sections": ["forecast", "policy", "limitations"]}))
    assert core.status("alice", run["id"])["state"] == "SAFETY_ACCEPTED"


def test_revocation_between_review_and_publish(core, run):
    reviewed(core, run["id"])
    core.revoke_version(TEMPLATE_VERSION)
    with pytest.raises(GateError, match="VERSION_REVOKED"):
        core.release("alice", run["id"])
    assert count(core, "releases") == 0


def test_old_reviewer_does_not_approve_new_draft(core, run):
    core.execute_models("alice", run["id"])
    draft(core, run["id"])
    entered, proceed = threading.Event(), threading.Event()
    def delayed(request):
        entered.set()
        assert proceed.wait(2)
        return fixture_review(request)
    core._fixtures["medical"] = replace(core._fixtures["medical"], call=delayed)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(core.review, "alice", run["id"], "medical")
        assert entered.wait(2)
        try:
            draft(core, run["id"], ["limitations", "policy", "forecast"])
        finally:
            proceed.set()
        with pytest.raises(GateError, match="STALE_ATTEMPT|STALE_REVIEW"):
            pending.result()
    assert core.status("alice", run["id"])["state"] == "DRAFT_READY"
    assert count(core, "reviews") == 0


def test_cross_run_capability_and_fenced_proposal(core, run, snapshot):
    core.execute_models("alice", run["id"])
    first = core.create_draft_job("alice", run["id"])
    other_case = core.create_case("alice", snapshot)
    other = core.start_run("alice", other_case["case_id"], "eval", "other-run")
    core.execute_models("alice", other["id"])
    second = core.create_draft_job("alice", other["id"])
    raw = canonical({"sections": ["forecast", "policy", "limitations"]})
    with pytest.raises(GateError, match="INVALID_CAPABILITY"):
        core.submit_proposal(second["job_id"], first["capability"], raw)
    core.create_draft_job("alice", run["id"])
    with pytest.raises(GateError, match="STALE_ATTEMPT"):
        core.submit_proposal(first["job_id"], first["capability"], raw)


def test_expired_proposal_and_reviewer_timeout_block_release(core, run):
    core.execute_models("alice", run["id"])
    job = core.create_draft_job("alice", run["id"])
    original_clock = core.clock
    core.clock = lambda: original_clock() + 301
    with pytest.raises(GateError, match="JOB_EXPIRED"):
        core.submit_proposal(job["job_id"], job["capability"], canonical({"sections": ["forecast", "policy", "limitations"]}))
    core.clock = original_clock
    draft(core, run["id"])
    def timeout(_):
        raise GateError("EXECUTOR_TIMEOUT")
    core._fixtures["ethics"] = replace(core._fixtures["ethics"], call=timeout)
    core.review("alice", run["id"], "medical")
    with pytest.raises(GateError, match="EXECUTOR_TIMEOUT"):
        core.review("alice", run["id"], "ethics")
    with pytest.raises(GateError, match="REVIEW_REQUIRED"):
        core.release("alice", run["id"])


def test_version_change_cannot_silently_reuse_weights(core, run):
    core._fixtures["prediction"] = replace(core._fixtures["prediction"], version="unregistered-v2")
    with pytest.raises(GateError, match="EXECUTOR_VERSION_MISMATCH"):
        core.execute_models("alice", run["id"])
    assert count(core, "artifacts") == 0
