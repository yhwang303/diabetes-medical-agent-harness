"""Trusted reviewer adapter: separate SDK sessions, staged verdict, Core-only registration."""

import fcntl
import json

from pydantic import ValidationError

from .contracts import GateError, ReviewOutput, digest
from .flash_gateway import FlashGateway
from .paths import confined
from .sdk_worker import MODEL, REVIEW_EXECUTORS
from .workers import WorkerLimits, WorkerRunner


class ReviewAgent:
    def __init__(self, key_path, data_dir, *, transport=None, runner=None, timeout=90):
        self.key_path, self.data_dir = confined(key_path), confined(data_dir)
        self.transport, self.timeout = transport, min(timeout, 90)
        self.runner = runner or WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
            file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)
        self.evidence = []

    def close(self):
        self.runner.close()

    def execute(self, core, executor, request, owner, run_id, job_id, role):
        if role not in REVIEW_EXECUTORS or executor != REVIEW_EXECUTORS[role]:
            raise GateError('INVALID_REVIEWER', 422)
        permission = core.data_permissions.authorize_run(owner, run_id, role + '_review')
        core.data_permissions.bind_job(owner, run_id, job_id, permission)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with confined(self.data_dir / 'review.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GateError('EXECUTOR_BUSY', 503) from None
            if self.key_path.stat().st_mode & 0o077:
                raise GateError('CREDENTIAL_PERMISSIONS', 503)
            key = self.key_path.read_text().strip()
            if not key:
                raise GateError('REVIEWER_NOT_CONFIGURED', 503)
            return self._execute(core, executor, request, owner, run_id, job_id, role, key)

    def _execute(self, core, executor, request, owner, run_id, job_id, role, key):
        staged, inspected = None, False
        trace = {'run_id': run_id, 'job_id': job_id, 'role': role, 'executor': executor.identity,
                 'version': executor.version, 'tools': [], 'workers': [], 'clinical_validation': False}
        self.evidence.append(trace)

        def validate():
            with core.store.tx() as db:
                run = core._run(db, run_id, owner)
                job = core._current_job(db, run, job_id)
                draft, body = core._draft(db, run)
                if (job['step'] != 'review_' + role or job['draft_id'] != draft['id']
                        or job['input_digest'] != digest(request) or draft['digest'] != request['report_hash']
                        or draft['evidence_hash'] != request['evidence_hash'] or body != request['report']):
                    raise GateError('REVIEW_BINDING')
                if core._review_request(db, run, draft, body, role) != request:
                    raise GateError('REVIEW_BINDING')
                if core._executor(run, role) != executor:
                    raise GateError('EXECUTOR_VERSION_MISMATCH')

        def audit(kind, **details):
            with core.store.tx() as db:
                core._run(db, run_id, owner)
                core._event(db, run_id, kind, job_id=job_id, role=role, **details)

        def notify(kind, **details):
            audit(kind, **details)
            trace['workers'].append({'kind': kind, **details})

        def operation(name, arguments):
            nonlocal staged, inspected
            try:
                validate()
                if name == 'inspect_review_packet':
                    if arguments != {}:
                        raise GateError('INVALID_TOOL_ARGUMENTS', 422)
                    # A fresh JSON copy prevents any caller from mutating the frozen packet.
                    result = {'ok': True, 'role': role, 'scope': 'engineering_only',
                              'packet': json.loads(json.dumps(request)), 'review_schema': ReviewOutput.model_json_schema()}
                else:
                    output = ReviewOutput.model_validate(arguments).model_dump()
                    if not inspected:
                        raise GateError('REVIEW_PACKET_REQUIRED')
                    if output['report_hash'] != request['report_hash'] or output['evidence_hash'] != request['evidence_hash']:
                        raise GateError('REVIEW_BINDING')
                    if output['verdict'] != 'pass' and not output['issues']:
                        raise GateError('REVIEW_REASON_REQUIRED')
                    if staged is not None:
                        raise GateError('REVIEW_ALREADY_STAGED')
                    result = {'ok': True, 'state': 'REVIEW_STAGED', 'clinical_use': False}
                audit('review_agent_tool', tool=name, accepted=True)
                if name == 'inspect_review_packet':
                    inspected = True
                else:
                    staged = output
            except (GateError, ValidationError) as exc:
                code = exc.code if isinstance(exc, GateError) else 'INVALID_REVIEW_OUTPUT'
                audit('review_agent_tool', tool=name, accepted=False, code=code)
                result = {'ok': False, 'error': code, 'clinical_use': False}
            trace['tools'].append({'name': name, 'ok': result['ok'], 'error': result.get('error')})
            return result

        try:
            with FlashGateway(key, self.data_dir / 'budget.json', None, transport=self.transport,
                tool_handlers={name: (lambda args, name=name: operation(name, args))
                               for name in ('inspect_review_packet', 'submit_review')}, authorize=validate) as gateway:
                trace['provider'] = gateway.records
                with core.data_lifecycle.session(owner, run_id, job_id) as session_id:
                    output = self.runner.run(executor,
                        {'port': gateway.port, 'capability': gateway.capability, 'scenario': 'review_' + role, 'session': session_id,
                         'review_config_hash': request['binding']['config_hash']},
                        timeout=self.timeout, validate=validate, job_id=job_id, notify=notify)
                trace['sdk'] = output
                results = output.get('results', [])
                if not results or results[-1].get('is_error') is not False or results[-1].get('terminal_reason') != 'completed':
                    raise GateError('REVIEW_AGENT_INCOMPLETE')
                if staged is None:
                    raise GateError('REVIEW_OUTPUT_MISSING')
                if any(r.get('error') for r in gateway.records):
                    raise GateError('REVIEW_PROVIDER_FAILED')
                if not any(r.get('model_responses') == [MODEL] for r in gateway.records):
                    raise GateError('REVIEW_PROVIDER_UNVERIFIED')
            validate()
            audit('review_agent_completed', executor=executor.identity, version=executor.version, model=MODEL,
                  session_id=results[-1].get('session_id'))
            return staged
        except Exception as exc:
            code = exc.code if isinstance(exc, GateError) else 'REVIEW_AGENT_FAILED'
            trace['error'] = code
            raise GateError(code, 503) from None
