"""One bounded desktop report task; the durable ledger owns status and cancellation."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from uuid import uuid4

from .contracts import AgentTaskRequest, GateError
from .basal_inputs import require_execution_profile
from .sdk_worker import MODEL, REPORT_OPERATIONS


ACTIVE = ('QUEUED', 'RUNNING', 'CANCELLING')


class AgentTasks:
    def __init__(self, core, reporter):
        self.core, self.reporter = core, reporter
        self._lock, self._closed = RLock(), False
        self._submitted = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='desktop-agent')
        # Called only by the exclusive service owner. Interrupted tasks never auto-resume billing.
        with core.store.tx() as db:
            rows = db.execute("SELECT * FROM agent_tasks WHERE state IN ('QUEUED','RUNNING','CANCELLING')").fetchall()
            for row in rows:
                core._invalidate(db, row['run_id'], 'SERVICE_RESTARTED')
                db.execute("UPDATE agent_tasks SET state='FAILED',reason='SERVICE_RESTARTED' WHERE id=?", (row['id'],))
                core._event(db, row['run_id'], 'agent_task_interrupted', task_id=row['id'])

    def configuration(self):
        key = self.reporter.key_path
        ready = key.is_file() and key.stat().st_size > 0 and not (key.stat().st_mode & 0o077)
        return {'enabled': True, 'credential_ready': ready, 'model': MODEL, 'task_kind': 'engineering_report',
                'max_concurrent': 1, 'clinical_use': False, 'tools': list(REPORT_OPERATIONS)}

    def start(self, owner, raw):
        request = AgentTaskRequest.model_validate(raw)
        core = self.core
        with self._lock:
            if self._closed:
                raise GateError('CORE_SHUTTING_DOWN', 503)
            with core.store.tx() as db:
                old = db.execute('SELECT * FROM agent_tasks WHERE owner=? AND idempotency_key=?',
                                 (owner, request.idempotency_key)).fetchone()
                if old:
                    if (old['case_id'], old['snapshot_id']) != (request.case_id, request.snapshot_id):
                        raise GateError('IDEMPOTENCY_CONFLICT')
                    old_id = old['id']
                else:
                    old_id = None
                    case = db.execute('SELECT * FROM cases WHERE id=? AND owner=?', (request.case_id, owner)).fetchone()
                    if not case:
                        raise GateError('NOT_FOUND', 404)
                    core.data_lifecycle.require_active(db, request.case_id)
                    if case['snapshot_id'] != request.snapshot_id:
                        raise GateError('SNAPSHOT_CHANGED')
                    snapshot = core._snapshot(db, request.snapshot_id)
                    require_execution_profile(snapshot)
                    if snapshot['source'] != 'synthetic':
                        raise GateError('FIXTURE_REQUIRES_SYNTHETIC_CASE')
                    if any(snapshot['missing_mask']):
                        raise GateError('MISSING_INPUT')
                    permission = core.data_permissions._require(db, owner, request.case_id, request.snapshot_id, 'report')
                    if db.execute("SELECT 1 FROM agent_tasks WHERE state IN ('QUEUED','RUNNING','CANCELLING')").fetchone():
                        raise GateError('EXECUTOR_BUSY', 503)
            if old_id:
                return self.status(owner, old_id)
            if not self.configuration()['credential_ready']:
                raise GateError('AGENT_NOT_CONFIGURED', 503)
            run = core.start_run(owner, request.case_id, 'eval', 'desktop-agent:' + request.idempotency_key,
                                 expected_snapshot_id=request.snapshot_id)
            task_id = uuid4().hex
            with core.store.tx() as db:
                core._live(db, core._run(db, run['id'], owner))
                db.execute('INSERT INTO agent_tasks VALUES(?,?,?,?,?,?,?,?,NULL,?)',
                           (task_id, owner, request.case_id, request.snapshot_id, run['id'], request.idempotency_key,
                            'QUEUED', 'evidence', core.clock()))
                core._event(db, run['id'], 'agent_task_created', task_id=task_id, task_kind='engineering_report')
            self._submitted = (task_id, self._pool.submit(self._work, owner, task_id, run['id'], permission))
            return self.status(owner, task_id)

    def _work(self, owner, task_id, run_id, permission):
        core = self.core
        state, reason = 'SUCCEEDED', None
        try:
            with core.store.tx() as db:
                core._live(db, core._run(db, run_id, owner))
                db.execute("UPDATE agent_tasks SET state='RUNNING' WHERE id=?", (task_id,))
            core.execute_models(owner, run_id)
            current = core.status(owner, run_id)
            if current['state'] != 'SAFETY_ACCEPTED' or not current['currently_valid']:
                raise GateError(current.get('reason') or 'EVIDENCE_REQUIRED')
            with core.store.tx() as db:
                core._live(db, core._run(db, run_id, owner))
                db.execute("UPDATE agent_tasks SET phase='report_agent' WHERE id=?", (task_id,))
            self.reporter.generate(owner, run_id, expected_permission=permission)
        except Exception as exc:
            state, reason = 'FAILED', exc.code if isinstance(exc, GateError) else 'AGENT_TASK_FAILED'
        finally:
            # Serialize final acknowledgement with cancellation; a late cancel must not leave CANCELLING forever.
            with self._lock, core.store.tx() as db:
                run = core._run(db, run_id, owner)
                if run['state'] == 'CANCELLED':
                    state, reason = 'CANCELLED', 'CANCELLED'
                elif state == 'SUCCEEDED':
                    try:
                        core._live(db, run)
                        core._draft(db, run)
                    except GateError as exc:
                        state, reason = 'FAILED', exc.code
                db.execute('UPDATE agent_tasks SET state=?,reason=? WHERE id=?', (state, reason, task_id))
                core._event(db, run_id, 'agent_task_finished', task_id=task_id, state=state, reason=reason)

    def status(self, owner, task_id):
        core = self.core
        with core.store.tx() as db:
            row = db.execute('SELECT * FROM agent_tasks WHERE id=? AND owner=?', (task_id, owner)).fetchone()
            if not row:
                raise GateError('NOT_FOUND', 404)
            if self._submitted and self._submitted[0] == task_id:
                future = self._submitted[1]
                if future.done() and future.exception() is not None:
                    raise GateError('STORAGE_UNAVAILABLE', 503)
            result = {key: row[key] for key in ('id', 'case_id', 'snapshot_id', 'run_id', 'state', 'phase', 'reason')}
            run = core._run(db, row['run_id'], owner)
            valid, invalid = True, None
            try:
                core._live(db, run)
                if row['state'] == 'SUCCEEDED':
                    core._draft(db, run)
            except GateError as exc:
                valid, invalid = False, exc.code
            events = db.execute("SELECT sequence,kind,body FROM events WHERE run_id=? AND kind IN "
                "('report_agent_tool','report_agent_completed','worker_started','worker_stopped') ORDER BY sequence DESC LIMIT 30",
                (row['run_id'],)).fetchall()
            # Project only fixed metadata. Never forward raw events, arguments, SDK text or report bodies.
            progress = []
            for event in reversed(events):
                body = json.loads(event['body'])
                item = {'sequence': event['sequence'], 'kind': event['kind']}
                item.update({key: body[key] for key in ('tool', 'accepted', 'code', 'session_id') if key in body})
                progress.append(item)
            return dict(result, run_state=run['state'], currently_valid=valid, invalid_reason=invalid,
                        active=row['state'] in ACTIVE, progress=progress, model=MODEL, clinical_use=False,
                        checked_at=core.clock())

    def latest(self, owner, case_id):
        with self.core.store.tx() as db:
            if not db.execute('SELECT 1 FROM cases WHERE id=? AND owner=?', (case_id, owner)).fetchone():
                raise GateError('NOT_FOUND', 404)
            row = db.execute('SELECT id FROM agent_tasks WHERE case_id=? AND owner=? ORDER BY rowid DESC LIMIT 1',
                             (case_id, owner)).fetchone()
        return self.status(owner, row['id']) if row else None

    def cancel(self, owner, task_id):
        with self._lock:
            status = self.status(owner, task_id)
            if status['active']:
                # Core fences all capabilities before the UI can acknowledge cancellation.
                self.core.cancel(owner, status['run_id'])
                with self.core.store.tx() as db:
                    db.execute("UPDATE agent_tasks SET state='CANCELLING',reason='CANCELLED' WHERE id=? AND state IN ('QUEUED','RUNNING')", (task_id,))
                    self.core._event(db, status['run_id'], 'agent_task_cancel_requested', task_id=task_id)
            return self.status(owner, task_id)

    def close(self):
        try:
            with self._lock:
                self._closed = True
                try:
                    with self.core.store.tx() as db:
                        rows = db.execute("SELECT owner,run_id FROM agent_tasks WHERE state IN ('QUEUED','RUNNING','CANCELLING')").fetchall()
                    for row in rows:
                        self.core.cancel(row['owner'], row['run_id'])
                finally:
                    self.reporter.runner.close()
        finally:
            self._pool.shutdown(wait=True)
