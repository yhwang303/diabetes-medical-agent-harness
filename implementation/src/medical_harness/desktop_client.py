"""Private desktop bridge client. Fixed localhost API operations; token never reaches the WebView."""

import json
import re
import sys
import time
from typing import get_args
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .contracts import FixtureScenario, GateError, canonical, strict_json
from .paths import ROOT


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def call(port, operation, payload):
    if not isinstance(payload, dict):
        raise GateError("DESKTOP_OPERATION_FORBIDDEN")
    if operation == "cases" and payload == {}:
        method, path, body = "GET", "/cases", None
    elif operation in {'mcp-status', 'mcp-check'} and payload == {}:
        method, path, body = ('GET', '/mcp/status', None) if operation == 'mcp-status' else ('POST', '/mcp/check', None)
    elif operation == 'case-lifecycle' and set(payload) == {'case_id'} and isinstance(payload['case_id'], str) and re.fullmatch(r'[0-9a-f]{32}', payload['case_id']):
        method, path, body = 'GET', f"/cases/{payload['case_id']}/lifecycle", None
    elif operation == 'case-delete':
        from .data_lifecycle import DeleteCaseRequest
        if set(payload) != {'case_id', 'snapshot_id', 'contract_version'} or not isinstance(payload['case_id'], str) or not re.fullmatch(r'[0-9a-f]{32}', payload['case_id']):
            raise GateError('DESKTOP_OPERATION_FORBIDDEN')
        request = DeleteCaseRequest.model_validate({k: v for k, v in payload.items() if k != 'case_id'})
        method, path, body = 'DELETE', f"/cases/{payload['case_id']}", canonical(request.model_dump()).encode()
    elif operation == 'data-permissions' and set(payload) == {'case_id'} and re.fullmatch(r'[0-9a-f]{32}', payload['case_id']):
        method, path, body = 'GET', f"/cases/{payload['case_id']}/data-permissions", None
    elif operation == 'data-permission-set':
        from .data_policy import DataPermissionRequest
        if set(payload) != {'case_id', 'snapshot_id', 'purpose', 'allowed', 'policy_version'} or not re.fullmatch(r'[0-9a-f]{32}', payload['case_id']):
            raise GateError('DESKTOP_OPERATION_FORBIDDEN')
        request = DataPermissionRequest.model_validate({k: v for k, v in payload.items() if k != 'case_id'})
        method, path, body = 'PUT', f"/cases/{payload['case_id']}/data-permissions", canonical(request.model_dump()).encode()
    elif operation == 'agent-configuration' and payload == {}:
        method, path, body = 'GET', '/agent/configuration', None
    elif operation == 'agent-start':
        from .contracts import AgentTaskRequest
        payload = AgentTaskRequest.model_validate(payload).model_dump()
        method, path, body = 'POST', '/agent/tasks', canonical(payload).encode()
    elif operation in {'agent-status', 'agent-cancel', 'agent-latest'}:
        key = 'case_id' if operation == 'agent-latest' else 'task_id'
        if set(payload) != {key} or not isinstance(payload[key], str) or not re.fullmatch(r'[0-9a-f]{32}', payload[key]):
            raise GateError('DESKTOP_OPERATION_FORBIDDEN')
        method = 'POST' if operation == 'agent-cancel' else 'GET'
        path = (f"/cases/{payload[key]}/agent-task" if operation == 'agent-latest' else
                f"/agent/tasks/{payload[key]}" + ('/cancel' if operation == 'agent-cancel' else ''))
        body = None
    elif operation in {"timeline", "evidence"} and set(payload) == {"case_id"} and re.fullmatch(r"[0-9a-f]{32}", payload["case_id"]):
        method, path, body = "GET", f"/cases/{payload['case_id']}/{operation}", None
    elif operation == "import" and set(payload) == {"file_name", "content"}:
        method, path, body = "POST", "/imports", canonical(payload).encode()
    elif operation in {"prediction", "rl-demo"} and set(payload) == (
            {"case_id", "snapshot_id", "idempotency_key"} | ({"scenario"} if operation == "rl-demo" else set())):
        if not all(isinstance(payload[k], str) and re.fullmatch(r"[0-9a-f]{32}", payload[k])
                   for k in ("case_id", "snapshot_id", "idempotency_key")):
            raise GateError("DESKTOP_OPERATION_FORBIDDEN")
        if operation == "rl-demo" and payload["scenario"] not in get_args(FixtureScenario):
            raise GateError("DESKTOP_OPERATION_FORBIDDEN")
        request = {"case_id": payload["case_id"], "mode": "eval",
                       "expected_snapshot_id": payload["snapshot_id"],
                       "idempotency_key": "desktop-" + operation + ":" + payload["idempotency_key"]}
        if operation == "rl-demo":
            request["fixture_scenario"] = payload["scenario"]
        run = _request(port, "POST", "/runs", canonical(request).encode())
        if not isinstance(run.get("id"), str) or not re.fullmatch(r"[0-9a-f]{32}", run["id"]):
            raise GateError("CORE_UNAVAILABLE")
        step = "execute" if operation == "rl-demo" else "prediction"
        return _request(port, "POST", f"/runs/{run['id']}/{step}", None)
    else:
        raise GateError("DESKTOP_OPERATION_FORBIDDEN")
    return _request(port, method, path, body)


def _request(port, method, path, body):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    token = (ROOT / "runtime/desktop-core/operator.token").read_text().strip()
    request = Request(f"http://127.0.0.1:{port}" + path, data=body, method=method,
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        # Core bounds executor waiting at five seconds; the client must allow its denial response.
        with opener.open(request, timeout=35 if path == '/mcp/check' else 12 if path.endswith("/execute") else 8 if path.endswith("/prediction") else 3) as response:
            return json.load(response)
    except HTTPError as exc:
        code = json.load(exc).get("error", "CORE_UNAVAILABLE")
        raise GateError(code) from None


def main():
    try:
        port, operation = int(sys.argv[1]), sys.argv[2]
        if not 1 <= port <= 65535:
            raise ValueError("port")
        payload = strict_json(sys.stdin.buffer.read(131073))
        # Only a read may wait for the app-owned service to start. Do not replay writes.
        for attempt in range(20):
            try:
                result = call(port, operation, payload)
                print(canonical({"ok": True, "data": result}))
                return
            except OSError:
                if operation != "cases" or attempt == 19:
                    raise
                time.sleep(0.1)
    except GateError as exc:
        print(canonical({"ok": False, "error": exc.code}))
    except Exception:
        print(canonical({"ok": False, "error": "CORE_UNAVAILABLE"}))


if __name__ == "__main__":
    main()
