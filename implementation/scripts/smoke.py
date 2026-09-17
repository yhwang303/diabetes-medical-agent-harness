"""Exercise a real service process and CLI, then stop it. All outputs stay local."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from medical_harness.cli import api_call

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime/smoke"


def start(port, log):
    process = subprocess.Popen([sys.executable, "-m", "medical_harness.cli", "serve", "--enable-fixtures",
                                "--port", str(port), "--data-dir", str(DATA)],
                               cwd=ROOT, stdout=log, stderr=log)
    url = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("service exited; inspect runtime/smoke/service.log")
        try:
            with opener.open(url + "/health", timeout=0.2) as response:
                health = json.load(response)
            assert health["fixtures_enabled"] and not health["clinical_use"]
            return process, url
        except OSError:
            time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("service startup timed out")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    os.umask(0o077)
    DATA.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = None
    with (DATA / "service.log").open("a") as log:
        try:
            process, url = start(port, log)
            token_file = DATA / "operator.token"
            token = token_file.read_text().strip()
            cli = [sys.executable, "-m", "medical_harness.cli", "--url", url, "--token-file", str(token_file)]
            def command(*args):
                output = subprocess.run(cli + list(args), cwd=ROOT, capture_output=True, text=True, check=True)
                return json.loads(output.stdout)

            # Stepwise CLI must use the same API gates as the full demo.
            case = command("case-create", str(ROOT / "examples/synthetic-case.json"))
            key = uuid4().hex
            created = command("run-start", case["case_id"], "--mode", "eval", "--idempotency-key", key)
            replay = command("run-start", case["case_id"], "--mode", "eval", "--idempotency-key", key)
            assert replay["id"] == created["id"]
            assert command("execute", created["id"])["state"] == "SAFETY_ACCEPTED"
            assert command("cancel", created["id"])["state"] == "CANCELLED"
            assert command("status", created["id"])["currently_valid"] is False
            rejected = subprocess.run(cli + ["execute", created["id"]], cwd=ROOT, capture_output=True, text=True)
            assert rejected.returncode == 1 and "CANCELLED" in rejected.stderr
            assert command("events", created["id"])[-1]["kind"] == "request_denied"
            result = subprocess.run(cli + ["demo"], cwd=ROOT, capture_output=True, text=True, check=True)
            bundle = json.loads(result.stdout)
            assert bundle["report"]["origin"] == "fixture"
            assert len(bundle["review_ids"]) == 2
            (DATA / "fixture-report.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
            run_id, release_id = bundle["run_id"], bundle["release_id"]
            events = api_call(url, token, "GET", f"/runs/{run_id}/events")
            assert any(e["kind"] == "release_committed" for e in events)
            (DATA / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))

            # The same server has fixtures available, but research mode still cannot use them.
            blocked = command("run-start", case["case_id"], "--idempotency-key", uuid4().hex)
            assert blocked["mode"] == "research"
            rejected = subprocess.run(cli + ["execute", blocked["id"]], cwd=ROOT, capture_output=True, text=True)
            assert rejected.returncode == 1 and "MODEL_NOT_CONFIGURED" in rejected.stderr
            blocked_status = api_call(url, token, "GET", f"/runs/{blocked['id']}")
            assert blocked_status["state"] == "UNAVAILABLE"

            stop(process)
            process = None
            process, url = start(port, log)
            restored = command("report", release_id)
            assert restored == bundle
            status = subprocess.run(cli + ["status", run_id], cwd=ROOT, capture_output=True, text=True, check=True)
            assert json.loads(status.stdout)["state"] == "RELEASED"
            evidence = {"run_id": run_id, "release_id": release_id, "report_hash": bundle["report_hash"],
                        "checks": {"real_http_service": True, "cli_fixture_chain": True,
                                   "two_reviewers": True, "research_rejects_fixture": True,
                                   "restart_report_parity": True, "cli_status": True,
                                   "cli_case_create_start_execute": True, "cli_start_idempotency": True,
                                   "cli_cancel_fences_execute": True, "cli_default_research": True,
                                   "cli_events_and_report": True},
                        "event_count": len(events), "blocked_run": blocked_status,
                        "clinical_validation": False, "real_models_connected": False}
            (DATA / "smoke-result.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
            print(json.dumps(evidence, ensure_ascii=False, indent=2))
        finally:
            if process is not None:
                stop(process)


if __name__ == "__main__":
    main()
