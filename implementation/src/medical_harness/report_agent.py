"""Host-side report orchestration. SDK proposes; Core validates and renders after completion."""

import fcntl

from pydantic import ValidationError

from .contracts import GateError, Proposal, canonical, digest
from .flash_gateway import FlashGateway
from .paths import confined
from .sdk_worker import MODEL, REPORT_EXECUTOR, REPORT_OPERATIONS
from .workers import WorkerLimits, WorkerRunner


class ReportAgent:
    def __init__(self, core, key_path, data_dir, *, transport=None, runner=None, timeout=90):
        self.core = core
        self.key_path, self.data_dir = confined(key_path), confined(data_dir)
        self.transport, self.timeout = transport, min(timeout, 90)
        self.runner = runner or WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
            file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)
        self.evidence = {}

    def generate(self, owner, run_id, *, expected_permission=None):
        permission = self.core.data_permissions.authorize_run(owner, run_id, 'report')
        if expected_permission is not None and permission != expected_permission:
            raise GateError('DATA_PERMISSION_CHANGED', 403)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A shared budget cannot be raced by separate service/CLI invocations.
        with confined(self.data_dir / 'report.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GateError('EXECUTOR_BUSY', 503) from exc
            if self.key_path.stat().st_mode & 0o077:
                raise GateError('CREDENTIAL_PERMISSIONS', 503)
            key = self.key_path.read_text().strip()
            if not key:
                raise GateError('MODEL_NOT_CONFIGURED', 503)
            return self._generate(owner, run_id, key, permission)

    def _generate(self, owner, run_id, key, permission):
        core = self.core
        draft = core.create_draft_job(owner, run_id)
        job_id = draft['job_id']
        staged = None
        self.evidence = {'run_id': run_id, 'job_id': job_id, 'tools': [], 'workers': [], 'clinical_use': False}

        def validate():
            with core.store.tx() as db:
                run = core._run(db, run_id, owner)
                core._current_job(db, run, job_id)
                if run['state'] != 'DRAFT_PENDING':
                    raise GateError('INVALID_STAGE')
                _, _, manifest = core._evidence(db, run)
                if digest(manifest) != draft['evidence_hash']:
                    raise GateError('EVIDENCE_CHANGED')

        def notify(kind, **details):
            with core.store.tx() as db:
                core._run(db, run_id, owner)
                core._event(db, run_id, kind, job_id=job_id, **details)
            self.evidence['workers'].append({'kind': kind, **details})

        def operation(name, arguments):
            nonlocal staged
            try:
                validate()
                if name != 'submit_report_proposal' and arguments != {}:
                    raise GateError('INVALID_TOOL_ARGUMENTS', 422)
                if name == 'inspect_report_contract':
                    result = {'ok': True, 'report_type': 'EngineeringFixtureReport', 'origin': 'fixture',
                              'clinical_use': False, 'template_version': draft['template_version'],
                              'evidence_hash': draft['evidence_hash'], 'proposal_schema': Proposal.model_json_schema()}
                elif name == 'inspect_report_status':
                    result = {'ok': True, 'state': 'DRAFT_PENDING', 'proposal_staged': staged is not None,
                              'review_required': True, 'clinical_use': False}
                else:
                    proposal = Proposal.model_validate(arguments).model_dump()
                    if staged is not None:
                        raise GateError('PROPOSAL_ALREADY_STAGED')
                    result = {'ok': True, 'state': 'PROPOSAL_STAGED', 'review_required': True, 'clinical_use': False}
                # Audit before accepting even an in-memory proposal. Never log arbitrary arguments.
                with core.store.tx() as db:
                    run = core._run(db, run_id, owner)
                    core._current_job(db, run, job_id)
                    core._event(db, run_id, 'report_agent_tool', job_id=job_id, tool=name, accepted=True)
                if name == 'submit_report_proposal':
                    staged = proposal
            except (GateError, ValidationError) as exc:
                code = exc.code if isinstance(exc, GateError) else 'INVALID_PROPOSAL'
                with core.store.tx() as db:
                    core._run(db, run_id, owner)
                    core._event(db, run_id, 'report_agent_tool', job_id=job_id, tool=name, accepted=False, code=code)
                result = {'ok': False, 'error': code, 'clinical_use': False}
            self.evidence['tools'].append({'name': name, 'ok': result['ok'], 'error': result.get('error')})
            return result

        try:
            core.data_permissions.bind_job(owner, run_id, job_id, permission)
            handlers = {name: (lambda args, name=name: operation(name, args)) for name in REPORT_OPERATIONS}
            with FlashGateway(key, self.data_dir / 'budget.json', None, transport=self.transport,
                              tool_handlers=handlers, authorize=validate) as gateway:
                self.evidence['provider'] = gateway.records
                with core.data_lifecycle.session(owner, run_id, job_id) as session_id:
                    output = self.runner.run(REPORT_EXECUTOR,
                        {'port': gateway.port, 'capability': gateway.capability, 'scenario': 'report', 'session': session_id},
                        timeout=self.timeout, validate=validate, job_id=job_id, notify=notify)
                self.evidence['sdk'] = output
                results = output.get('results', [])
                if not results or results[-1].get('is_error') is not False or results[-1].get('terminal_reason') != 'completed':
                    raise GateError('REPORT_AGENT_INCOMPLETE')
                if staged is None:
                    raise GateError('REPORT_PROPOSAL_MISSING')
                # Completion/error status is authoritative to this host, never inferred from model prose.
                if any(r.get('error') for r in gateway.records):
                    raise GateError('REPORT_PROVIDER_FAILED')
                if not any(r.get('model_responses') == [MODEL] for r in gateway.records):
                    raise GateError('REPORT_PROVIDER_UNVERIFIED')
            validate()
            with core.store.tx() as db:
                core._event(db, run_id, 'report_agent_completed', job_id=job_id,
                            executor=REPORT_EXECUTOR.identity, version=REPORT_EXECUTOR.version, model=MODEL,
                            session_id=results[-1].get('session_id'))
            result = core.submit_proposal(job_id, draft['capability'], canonical(staged))
            self.evidence['result'] = result
            return result
        except Exception as exc:
            code = exc.code if isinstance(exc, GateError) else 'REPORT_AGENT_FAILED'
            self.evidence['error'] = code
            core._job_failure(owner, run_id, job_id, code)
            raise GateError(code, 503) from None
