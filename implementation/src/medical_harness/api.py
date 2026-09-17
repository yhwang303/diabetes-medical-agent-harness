"""Authenticated localhost facade. No trusted-result registration endpoint exists."""

import secrets
import sqlite3

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .contracts import ErrorResponse, GateError, RunRequest, contract_catalog, strict_json
from .core import Core
from .mcp_diagnostics import McpDiagnostics


def error_response(code, status):
    return JSONResponse(ErrorResponse(error=code).model_dump(), status_code=status)


def create_app(core: Core, tokens: dict[str, str], *, agent_tasks=None, prediction_agent=None, rl_agent=None) -> FastAPI:
    app = FastAPI(title="Medical Research Harness", version="0.1.0", docs_url=None, redoc_url=None,
                  description="工程测试 Harness。真实预测、RL、医学审核尚未配置。")
    diagnostics = McpDiagnostics(core, prediction_enabled=prediction_agent is not None, rl_enabled=rl_agent is not None)
    app.state.mcp_diagnostics = diagnostics
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    def bearer(request):
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not value or len(value) > 256 or not value.isascii():
            raise GateError("AUTH_REQUIRED", 401)
        return value

    def owner(request):
        value = bearer(request)
        for token, identity in tokens.items():
            if secrets.compare_digest(value, token):
                request.state.owner = identity
                return identity
        raise GateError("AUTH_REQUIRED", 401)

    async def raw_body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 131072:
                raise GateError("PAYLOAD_TOO_LARGE", 413)
        return bytes(data)

    async def body(request):
        return strict_json(await raw_body(request))

    @app.middleware("http")
    async def browser_boundary(request, call_next):
        # API is a bearer-authenticated local service, not a credentialed web origin.
        # Block browser-origin calls, including DNS-rebinding attempts and null origins.
        if "origin" in request.headers:
            response = error_response("BROWSER_ORIGIN_FORBIDDEN", 403)
        else:
            response = await call_next(request)
            # TrustedHostMiddleware's transport rejection otherwise returns plain text.
            if response.status_code == 400 and response.headers.get("content-type", "").startswith("text/plain"):
                response = error_response("INVALID_HOST", 400)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        return response

    @app.exception_handler(GateError)
    async def gate_error(request, exc):
        identity = getattr(request.state, "owner", None)
        run_id = request.path_params.get("run_id")
        if identity:
            try:
                await run_in_threadpool(core.record_denial, identity, run_id, exc.code)
            except GateError:
                pass  # Unknown/cross-owner IDs must not write into that run's history.
            except sqlite3.Error:
                return error_response("AUDIT_UNAVAILABLE", 503)
        return error_response(exc.code, exc.status)

    @app.exception_handler(ValidationError)
    async def invalid_schema(request, exc):
        # Pydantic's default error can echo untrusted input. Return fixed metadata only.
        return await gate_error(request, GateError("INVALID_SCHEMA", 422))

    @app.exception_handler(sqlite3.Error)
    async def storage_failure(request, exc):
        return error_response("STORAGE_UNAVAILABLE", 503)

    @app.exception_handler(HTTPException)
    async def transport_error(request, exc):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        response = error_response(code, exc.status_code)
        response.headers.update(exc.headers or {})
        return response

    @app.get("/contracts")
    async def contracts():
        return contract_catalog()

    @app.get("/health")
    async def health():
        return {"service": "medical-research-harness", "fixtures_enabled": core.enable_fixtures,
                "real_models_configured": False, "clinical_use": False}

    @app.get('/mcp/status')
    async def mcp_status(request: Request):
        return await run_in_threadpool(diagnostics.status, owner(request))

    @app.post('/mcp/check')
    async def mcp_check(request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError('CLIENT_BODY_FORBIDDEN', 422)
        return await run_in_threadpool(diagnostics.check, identity)

    def tasks():
        if agent_tasks is None:
            raise GateError('AGENT_NOT_CONFIGURED', 503)
        return agent_tasks

    @app.get('/agent/configuration')
    async def agent_configuration(request: Request):
        owner(request)
        return tasks().configuration()

    @app.post('/agent/tasks', status_code=202)
    async def agent_start(request: Request):
        identity = owner(request)
        return await run_in_threadpool(tasks().start, identity, await body(request))

    @app.get('/agent/tasks/{task_id}')
    async def agent_status(task_id: str, request: Request):
        identity = owner(request)
        return await run_in_threadpool(tasks().status, identity, task_id)

    @app.get('/cases/{case_id}/agent-task')
    async def agent_latest(case_id: str, request: Request):
        identity = owner(request)
        return await run_in_threadpool(tasks().latest, identity, case_id)

    @app.post('/agent/tasks/{task_id}/cancel')
    async def agent_cancel(task_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError('CLIENT_BODY_FORBIDDEN', 422)
        return await run_in_threadpool(tasks().cancel, identity, task_id)

    @app.post("/cases", status_code=201)
    async def create_case(request: Request):
        identity = owner(request)
        return await run_in_threadpool(core.create_case, identity, await body(request))

    @app.post("/imports", status_code=201)
    async def import_case(request: Request):
        identity = owner(request)
        return await run_in_threadpool(core.import_case, identity, await body(request))

    @app.get("/cases")
    async def list_cases(request: Request):
        return await run_in_threadpool(core.list_cases, owner(request))

    @app.get('/cases/{case_id}/lifecycle')
    async def case_lifecycle(case_id: str, request: Request):
        return await run_in_threadpool(core.data_lifecycle.read, owner(request), case_id)

    @app.delete('/cases/{case_id}')
    async def delete_case(case_id: str, request: Request):
        identity = owner(request)
        return await run_in_threadpool(core.data_lifecycle.delete, identity, case_id, await body(request))

    @app.get('/cases/{case_id}/data-permissions')
    async def data_permissions(case_id: str, request: Request):
        return await run_in_threadpool(core.data_permissions.read, owner(request), case_id)

    @app.put('/cases/{case_id}/data-permissions')
    async def set_data_permission(case_id: str, request: Request):
        identity = owner(request)
        return await run_in_threadpool(core.data_permissions.update, identity, case_id, await body(request))

    @app.get("/cases/{case_id}/timeline")
    async def input_timeline(case_id: str, request: Request):
        return await run_in_threadpool(core.input_timeline, owner(request), case_id)

    @app.get("/cases/{case_id}/evidence")
    async def case_evidence(case_id: str, request: Request):
        return await run_in_threadpool(core.case_evidence, owner(request), case_id)

    @app.put("/cases/{case_id}/snapshot")
    async def update_case(case_id: str, request: Request):
        identity = owner(request)
        return await run_in_threadpool(core.update_case, identity, case_id, await body(request))

    @app.post("/runs", status_code=201)
    async def start_run(request: Request):
        identity = owner(request)
        data = RunRequest.model_validate(await body(request))
        return await run_in_threadpool(core.start_run, identity, data.case_id, data.mode, data.idempotency_key,
                                      data.expected_snapshot_id, data.fixture_scenario)

    @app.get("/runs/{run_id}")
    async def run_status(run_id: str, request: Request):
        return await run_in_threadpool(core.status, owner(request), run_id)

    @app.post('/runs/{run_id}/prediction-agent')
    async def prediction_agent_run(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError('CLIENT_BODY_FORBIDDEN', 422)
        if prediction_agent is None:
            raise GateError('AGENT_NOT_CONFIGURED', 503)
        return await run_in_threadpool(prediction_agent.run, identity, run_id)

    @app.post('/runs/{run_id}/rl-agent')
    async def rl_agent_run(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError('CLIENT_BODY_FORBIDDEN', 422)
        if rl_agent is None:
            raise GateError('AGENT_NOT_CONFIGURED', 503)
        return await run_in_threadpool(rl_agent.run, identity, run_id)

    @app.post("/runs/{run_id}/prediction")
    async def prediction(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_BODY_FORBIDDEN", 422)
        return await run_in_threadpool(core.execute_prediction, identity, run_id)

    @app.post("/runs/{run_id}/execute")
    async def execute(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_BODY_FORBIDDEN", 422)
        return await run_in_threadpool(core.execute_models, identity, run_id)

    @app.post("/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_BODY_FORBIDDEN", 422)
        return await run_in_threadpool(core.cancel, identity, run_id)

    @app.post("/runs/{run_id}/draft-jobs")
    async def draft_job(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_BODY_FORBIDDEN", 422)
        return await run_in_threadpool(core.create_draft_job, identity, run_id)

    @app.post("/proposal-jobs/{job_id}")
    async def proposal(job_id: str, request: Request):
        # Narrow one-shot capability. A researcher or another job's token cannot submit.
        capability = bearer(request)
        return await run_in_threadpool(core.submit_proposal, job_id, capability, await raw_body(request))

    @app.post("/runs/{run_id}/reviews/{role}")
    async def review(run_id: str, role: str, request: Request):
        # Request execution of a Core-assigned reviewer; no client verdict is accepted.
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_VERDICT_FORBIDDEN", 422)
        return await run_in_threadpool(core.review, identity, run_id, role)

    @app.post("/runs/{run_id}/release")
    async def release(run_id: str, request: Request):
        identity = owner(request)
        if await raw_body(request):
            raise GateError("CLIENT_RELEASE_BODY_FORBIDDEN", 422)
        return await run_in_threadpool(core.release, identity, run_id)

    @app.get("/releases/{release_id}")
    async def get_release(release_id: str, request: Request):
        return await run_in_threadpool(core.get_release, owner(request), release_id)

    @app.get("/runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        return await run_in_threadpool(core.events, owner(request), run_id)

    return app
