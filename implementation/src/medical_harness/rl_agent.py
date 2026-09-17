"""Run-scoped RL MCP tools. Only Core can execute and register numerical evidence."""

import fcntl
from typing import Annotated

from pydantic import Field, ValidationError

from .contracts import GateError, StrictModel, digest
from .flash_gateway import FlashGateway
from .paths import confined
from .sdk_worker import MODEL, RL_AGENT_EXECUTOR, rl_configuration
from .workers import WorkerLimits, WorkerRunner


class RLJobQuery(StrictModel):
    job_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]


def binding(core, db, run):
    _, prediction_hash = core._artifact(db, run, 'prediction', completed_controller=True)
    return digest({'configuration': rl_configuration(), 'prediction_hash': prediction_hash, 'run_id': run['id'],
        'case_id': run['case_id'], 'snapshot_id': run['snapshot_id'],
        'snapshot_hash': digest(core._snapshot(db, run['snapshot_id']))})


class RLTools:
    def __init__(self, core, owner, run_id, agent_job_id):
        self.core, self.owner, self.run_id, self.agent_job_id = core, owner, run_id, agent_job_id

    def validate(self, db):
        run = self.core._run(db, self.run_id, self.owner)
        job = self.core._current_job(db, run, self.agent_job_id)
        if not db.execute('SELECT 1 FROM job_data_permissions WHERE job_id=?', (self.agent_job_id,)).fetchone():
            raise GateError('DATA_PERMISSION_INTEGRITY', 403)
        if job['step'] != 'rl_agent' or job['input_digest'] != binding(self.core, db, run):
            raise GateError('RL_AGENT_BINDING')
        return run

    def inspect(self, db, run, job_id):
        job = db.execute('SELECT * FROM jobs WHERE id=? AND run_id=? AND step=?',
                         (job_id, run['id'], 'policy')).fetchone()
        if not job:
            raise GateError('NOT_FOUND', 404)
        dependency = db.execute('SELECT parent_job_id FROM job_dependencies WHERE child_job_id=?', (job_id,)).fetchone()
        if not dependency or dependency['parent_job_id'] != self.agent_job_id:
            raise GateError('RL_JOB_BINDING')
        result = {'ok': True, 'job_id': job_id, 'status': job['status'],
                  'policy_available': False, 'clinical_use': False, 'safety_checked': False}
        if job['status'] == 'ACCEPTED':
            body, artifact_hash = self.core._artifact(db, run, 'policy')
            if body['job_id'] != job_id:
                raise GateError('RL_JOB_BINDING')
            result.update(policy_available=True, evidence={
                'artifact_id': body['id'], 'artifact_hash': artifact_hash, 'origin': body['origin'],
                'producer': body['producer'], 'version': body['version']})
        elif job['status'] != 'RUNNING':
            result['reason'] = run['reason'] or 'RL_UNAVAILABLE'
        return result

    def call(self, name, raw):
        core = self.core
        try:
            if name == 'request_rl':
                if raw != {}: raise GateError('INVALID_TOOL_ARGUMENTS', 422)
                with core.store.tx() as db:
                    run = self.validate(db)
                    old = db.execute("SELECT id FROM jobs WHERE run_id=? AND step='policy' ORDER BY attempt DESC LIMIT 1",
                                     (self.run_id,)).fetchone()
                    result = self.inspect(db, run, old['id']) if old else None
                if result is None:
                    # Synchronous tool, bounded by Core's numerical-worker deadline. No hidden retry.
                    core._model_step(self.owner, self.run_id, 'policy', authorization_job_id=self.agent_job_id)
                    with core.store.tx() as db:
                        run = self.validate(db)
                        job = db.execute("SELECT id FROM jobs WHERE run_id=? AND step='policy'", (self.run_id,)).fetchone()
                        if not job: raise GateError('RL_UNAVAILABLE')
                        result = self.inspect(db, run, job['id'])
            elif name == 'inspect_rl':
                query = RLJobQuery.model_validate(raw)
                with core.store.tx() as db:
                    run = self.validate(db)
                    result = self.inspect(db, run, query.job_id)
            else:
                raise GateError('SDK_TOOL_DENIED', 403)
        except (GateError, ValidationError) as exc:
            result = {'ok': False, 'error': exc.code if isinstance(exc, GateError) else 'INVALID_TOOL_ARGUMENTS',
                      'policy_available': False, 'clinical_use': False, 'safety_checked': False}
        # No arbitrary arguments or model prose enter the ledger.
        with core.store.tx() as db:
            core._run(db, self.run_id, self.owner)
            core._event(db, self.run_id, 'rl_agent_tool', job_id=self.agent_job_id,
                        tool=name if name in ('request_rl', 'inspect_rl') else 'denied',
                        accepted=result['ok'], code=result.get('error'))
        return result


class RLAgent:
    def __init__(self, core, key_path, data_dir, *, transport=None, runner=None, timeout=90):
        self.core = core
        self.key_path, self.data_dir = confined(key_path), confined(data_dir)
        self.transport, self.timeout = transport, min(timeout, 90)
        self.runner = runner or WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
            file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)
        self.evidence = {}

    def close(self):
        self.runner.close()

    def _receipt(self, db, run, agent_job_id):
        job = db.execute("SELECT id FROM jobs WHERE run_id=? AND step='policy'", (run['id'],)).fetchone()
        if not job: raise GateError('RL_REQUIRED')
        result = RLTools(self.core, run['owner'], run['id'], agent_job_id).inspect(db, run, job['id'])
        if not result['policy_available']: raise GateError(result.get('reason') or 'RL_REQUIRED')
        return dict(result, sdk_job_id=agent_job_id)

    def run(self, owner, run_id):
        core = self.core
        permission = core.data_permissions.authorize_run(owner, run_id, 'rl')
        with core.store.tx() as db:
            run = core._run(db, run_id, owner)
            old = db.execute("SELECT * FROM jobs WHERE run_id=? AND step='rl_agent'", (run_id,)).fetchone()
            if old:
                if old['status'] != 'ACCEPTED': raise GateError('RL_AGENT_ALREADY_STARTED')
                if old['input_digest'] != binding(core, db, run): raise GateError('RL_AGENT_BINDING')
                return self._receipt(db, run, old['id'])
            if run['state'] != 'PREDICTION_ACCEPTED': raise GateError('INVALID_STAGE')
            binding(core, db, run)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with confined(self.data_dir / 'rl.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise GateError('EXECUTOR_BUSY', 503) from None
            try:
                if self.key_path.stat().st_mode & 0o077: raise GateError('CREDENTIAL_PERMISSIONS', 503)
                key = self.key_path.read_text().strip()
            except OSError:
                raise GateError('MODEL_NOT_CONFIGURED', 503) from None
            if not key: raise GateError('MODEL_NOT_CONFIGURED', 503)
            with core.store.tx() as db:
                run = core._run(db, run_id, owner)
                core._live(db, run)
                if db.execute("SELECT 1 FROM jobs WHERE run_id=? AND step='rl_agent'", (run_id,)).fetchone():
                    raise GateError('RL_AGENT_ALREADY_STARTED')
                if run['state'] != 'PREDICTION_ACCEPTED': raise GateError('INVALID_STAGE')
                agent_job_id = core._new_job(db, run, 'rl_agent', binding(core, db, run))
                db.execute('UPDATE jobs SET deadline=? WHERE id=?',
                           (min(core.clock() + self.timeout, run['expires']), agent_job_id))
            self.evidence = {'run_id': run_id, 'job_id': agent_job_id, 'workers': [], 'clinical_use': False, 'safety_checked': False}
            tools = RLTools(core, owner, run_id, agent_job_id)
            def validate():
                with core.store.tx() as db: tools.validate(db)
            def notify(kind, **details):
                with core.store.tx() as db:
                    core._run(db, run_id, owner)
                    core._event(db, run_id, kind, job_id=agent_job_id, **details)
                self.evidence['workers'].append({'kind': kind, **details})
            try:
                core.data_permissions.bind_job(owner, run_id, agent_job_id, permission)
                with FlashGateway(key, self.data_dir / 'budget.json', None, transport=self.transport,
                    tool_handlers={name: (lambda args, name=name: tools.call(name, args))
                                   for name in ('request_rl', 'inspect_rl')}, authorize=validate) as gateway:
                    self.evidence['provider'] = gateway.records
                    with core.data_lifecycle.session(owner, run_id, agent_job_id) as session_id:
                        output = self.runner.run(RL_AGENT_EXECUTOR,
                            {'port': gateway.port, 'capability': gateway.capability, 'session': session_id,
                             'scenario': 'rl', 'rl_config_hash': digest(rl_configuration())},
                            timeout=self.timeout, validate=validate, job_id=agent_job_id, notify=notify)
                    self.evidence['sdk'] = output
                    results = output.get('results', [])
                    if not results or results[-1].get('is_error') is not False or results[-1].get('terminal_reason') != 'completed':
                        raise GateError('RL_AGENT_INCOMPLETE')
                    if any(r.get('error') for r in gateway.records): raise GateError('RL_PROVIDER_FAILED')
                    if not any(r.get('model_responses') == [MODEL] for r in gateway.records):
                        raise GateError('RL_PROVIDER_UNVERIFIED')
                with core.store.tx() as db:
                    run = tools.validate(db)
                    result = self._receipt(db, run, agent_job_id)
                    core._event(db, run_id, 'rl_agent_completed', job_id=agent_job_id,
                                executor=RL_AGENT_EXECUTOR.identity, configuration_hash=digest(rl_configuration()))
                    db.execute("UPDATE jobs SET status='ACCEPTED' WHERE id=?", (agent_job_id,))
                return result
            except Exception as exc:
                code = exc.code if isinstance(exc, GateError) else 'RL_AGENT_FAILED'
                self.evidence['error'] = code
                core._job_failure(owner, run_id, agent_job_id, code)
                with core.store.tx() as db:
                    run = core._run(db, run_id, owner)
                    if run['state'] not in ('CANCELLED', 'INVALIDATED'):
                        core._invalidate(db, run_id, code)
                raise GateError(code, 503) from None
