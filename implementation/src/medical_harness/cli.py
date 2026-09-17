"""Local service launcher and thin HTTP client; no CLI bypass of Core gates."""

import argparse
import fcntl
import json
import os
import secrets
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from .contracts import GateError, canonical, strict_json
from .paths import ROOT, confined


def api_call(url, token, method, path, body=None):
    request = Request(url + path, data=None if body is None else canonical(body).encode(), method=method,
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    # Local requests must not pass through configured external proxies.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        error = json.load(exc)
        raise RuntimeError(f"HTTP {exc.code}: {error.get('error', 'request failed')}") from None


def demo(url, token):
    snapshot = json.loads((ROOT / "examples/synthetic-case.json").read_text())
    case = api_call(url, token, "POST", "/cases", snapshot)
    run = api_call(url, token, "POST", "/runs", {"case_id": case["case_id"], "mode": "eval",
                                               "idempotency_key": uuid4().hex})
    run_id = run["id"]
    api_call(url, token, "POST", f"/runs/{run_id}/execute")
    job = api_call(url, token, "POST", f"/runs/{run_id}/draft-jobs")
    api_call(url, job["capability"], "POST", "/proposal-jobs/" + job["job_id"],
             {"sections": ["forecast", "policy", "limitations"]})
    for role in ["medical", "ethics"]:
        api_call(url, token, "POST", f"/runs/{run_id}/reviews/{role}")
    result = api_call(url, token, "POST", f"/runs/{run_id}/release")
    return api_call(url, token, "GET", "/releases/" + result["release_id"])


def main():
    parser = argparse.ArgumentParser(description="Medical research Harness; engineering fixtures only")
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--token-file", type=Path, default=ROOT / "runtime/operator.token")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--enable-fixtures", action="store_true")
    serve.add_argument('--enable-basal-fixtures', action='store_true', help='Opt in to synthetic basal contract evaluation; also requires --enable-fixtures')
    serve.add_argument("--enable-rl-agent", action="store_true", help="Opt in to run-scoped RL MCP sessions; no calls at startup")
    serve.add_argument("--enable-prediction-agent", action="store_true", help="Opt in to run-scoped prediction MCP sessions; no calls at startup")
    serve.add_argument("--enable-agent", action="store_true", help="Enable explicit desktop report tasks; no calls at startup")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--data-dir", type=Path, default=ROOT / "runtime")
    sub.add_parser("demo")
    create = sub.add_parser("case-create", help="Freeze a structured JSON snapshot through Core")
    create.add_argument("snapshot_file", type=Path)
    start = sub.add_parser("run-start", help="Create a run; defaults to research, never a fixture fallback")
    start.add_argument("case_id")
    start.add_argument("--mode", choices=["research", "eval"], default="research")
    start.add_argument("--idempotency-key", required=True)
    for command in ("execute", "cancel"):
        action = sub.add_parser(command)
        action.add_argument("run_id")
    status = sub.add_parser("status")
    status.add_argument("run_id")
    events = sub.add_parser("events")
    events.add_argument("run_id")
    release = sub.add_parser("report")
    release.add_argument("release_id")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        from .core import Core
        from .store import Store

        try:
            data = confined(args.data_dir)
            for name in ("service.lock", "operator.token", "harness.sqlite3"):
                confined(data / name)
        except ValueError as exc:
            parser.error(str(exc))
        os.umask(0o077)
        data.mkdir(parents=True, exist_ok=True)
        # One service owns this DB; startup recovery must not fence another process.
        with (data / "service.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another Harness service already owns this data directory")
            token_file = data / "operator.token"
            if not token_file.exists():
                token_file.write_text(secrets.token_urlsafe(32))
            token_file.chmod(0o600)
            token = token_file.read_text().strip()
            if len(token) < 32 or len(token) > 256 or not token.isascii() or any(c.isspace() for c in token):
                parser.error("invalid operator token file; service not started")
            core = Core(Store(data / "harness.sqlite3"), enable_fixtures=args.enable_fixtures,
                        enable_basal_fixtures=args.enable_basal_fixtures)
            core.recover_interrupted()
            core.data_lifecycle.recover()
            prediction_agent = None
            if args.enable_prediction_agent:
                from .prediction_agent import PredictionAgent
                prediction_agent = PredictionAgent(core, ROOT / 'runtime/private/deepseek.key', data / 'prediction-agent')
            rl_agent = None
            if args.enable_rl_agent:
                from .rl_agent import RLAgent
                rl_agent = RLAgent(core, ROOT / 'runtime/private/deepseek.key', data / 'rl-agent')
            agent_tasks = None
            if args.enable_agent:
                from .agent_tasks import AgentTasks
                from .report_agent import ReportAgent
                agent_tasks = AgentTasks(core, ReportAgent(core, ROOT / 'runtime/private/deepseek.key', data / 'agent'))
            print(f"Token file: {token_file} (credential not printed)", flush=True)
            try:
                uvicorn.run(create_app(core, {token: "local-researcher"}, agent_tasks=agent_tasks, prediction_agent=prediction_agent, rl_agent=rl_agent), host="127.0.0.1", port=args.port,
                            access_log=False, log_level="warning", timeout_graceful_shutdown=3)
            finally:
                try:
                    if agent_tasks is not None:
                        agent_tasks.close()
                finally:
                    if prediction_agent is not None:
                        prediction_agent.close()
                    if rl_agent is not None:
                        rl_agent.close()
                    core.close()
        return
    try:
        token = args.token_file.read_text().strip()
        if args.command == "demo":
            result = demo(args.url, token)
        elif args.command == "case-create":
            with args.snapshot_file.open("rb") as source:
                snapshot = strict_json(source.read(131073))
            result = api_call(args.url, token, "POST", "/cases", snapshot)
        elif args.command == "run-start":
            result = api_call(args.url, token, "POST", "/runs", {
                "case_id": args.case_id, "mode": args.mode, "idempotency_key": args.idempotency_key})
        elif args.command in {"execute", "cancel"}:
            result = api_call(args.url, token, "POST", f"/runs/{args.run_id}/{args.command}")
        else:
            path = (f"/runs/{args.run_id}" if args.command == "status" else f"/runs/{args.run_id}/events"
                    if args.command == "events" else f"/releases/{args.release_id}")
            result = api_call(args.url, token, "GET", path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (RuntimeError, OSError, GateError) as exc:
        print(exc.code if isinstance(exc, GateError) else str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
