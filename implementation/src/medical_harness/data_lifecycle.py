"""Local content deletion and host-owned SDK work directories; no remote erasure claim."""

from contextlib import contextmanager
import hashlib
import re
import shutil
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field

from .contracts import GateError, StrictModel
from .paths import ROOT

SESSION_ROOT = ROOT / 'runtime/sdk-smoke/sessions'


class DeleteCaseRequest(StrictModel):
    snapshot_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    contract_version: Literal['engineering-lifecycle-v1']


def lifecycle_catalog():
    return {'version': 'engineering-lifecycle-v1',
        'case_content': 'retained_until_explicit_owner_delete',
        'delete_scope': ['all_case_snapshots', 'import_metadata', 'artifacts', 'drafts', 'reviews',
                         'releases', 'run_profiles', 'review_configs', 'outbound_permissions'],
        'retained': ['ownership_and_deletion_receipt', 'run_job_task_identifiers_and_status',
                     'controlled_audit_metadata_and_hashes', 'aggregate_provider_budget'],
        'audit_retention': 'until_P16_retention_policy_no_automatic_expiry',
        'sdk': 'registered_prediction_rl_report_review_directories_removed_after_worker_returns',
        'sdk_recovery': 'exclusive_service_startup_retries_own_ledger_directories',
        'legacy_sdk': 'unmapped_historical_smoke_directories_not_automatically_deleted',
        'backups': 'no_automatic_backup_existing_copies_not_erased_by_case_delete',
        'restore': 'new_service_permission_required_deletion_receipts_preserved_if_present',
        'pre_deletion_backup': 'may_restore_content_requires_deletion_reconciliation_before_real_data',
        'external_provider': 'retention_and_deletion_unverified_local_withdrawal_is_not_remote_erasure',
        'physical_erasure': False, 'memory_erasure': False, 'real_personal_data_enabled': False}


class DataLifecycle:
    def __init__(self, core):
        self.core = core
        # A copied ledger must never clean directories belonging to the original path.
        self.scope = hashlib.sha256(str(core.store.path).encode()).hexdigest()

    def require_active(self, db, case_id):
        if db.execute('SELECT 1 FROM case_deletions WHERE case_id=?', (case_id,)).fetchone():
            raise GateError('CASE_DELETED', 410)
        if db.execute("SELECT 1 FROM sdk_sessions WHERE case_id=? AND state IN ('CLEANUP_FAILED','PATH_CONFLICT')", (case_id,)).fetchone():
            raise GateError('SDK_CLEANUP_REQUIRED', 503)

    def _case(self, db, owner, case_id):
        case = db.execute('SELECT * FROM cases WHERE id=? AND owner=?', (case_id, owner)).fetchone()
        if not case:
            raise GateError('NOT_FOUND', 404)
        return case

    def read(self, owner, case_id):
        with self.core.store.tx() as db:
            self._case(db, owner, case_id)
            deletion = db.execute('SELECT * FROM case_deletions WHERE case_id=?', (case_id,)).fetchone()
            sessions = db.execute("SELECT state,count(*) AS n FROM sdk_sessions WHERE case_id=? AND state!='REMOVED' GROUP BY state",
                                  (case_id,)).fetchall()
            pending = {row['state']: row['n'] for row in sessions}
            return {'case_id': case_id, 'state': ('SDK_CLEANUP_PENDING' if pending else 'LOCAL_CONTENT_REMOVED')
                    if deletion else 'RETAINED', 'local_content_removed': bool(deletion),
                    'requested_at': deletion['requested_at'] if deletion else None,
                    'sdk_pending': pending, 'contract': lifecycle_catalog()}

    def delete(self, owner, case_id, raw):
        request = DeleteCaseRequest.model_validate(raw)
        core = self.core
        with core.store.tx() as db:
            case = self._case(db, owner, case_id)
            if case['snapshot_id'] != request.snapshot_id:
                raise GateError('SNAPSHOT_CHANGED')
            if not db.execute('SELECT 1 FROM case_deletions WHERE case_id=?', (case_id,)).fetchone():
                db.execute('INSERT INTO case_deletions VALUES(?,?)', (case_id, core.clock()))
                # Retain small execution envelopes so in-flight finalizers can safely record termination.
                for row in db.execute('SELECT id FROM runs WHERE case_id=?', (case_id,)).fetchall():
                    run_id = row['id']
                    core._invalidate(db, run_id, 'CASE_DELETED')
                    db.execute('DELETE FROM job_data_permissions WHERE job_id IN (SELECT id FROM jobs WHERE run_id=?)', (run_id,))
                    for table in ('artifacts', 'drafts', 'reviews', 'releases', 'run_profiles', 'review_configs'):
                        db.execute(f'DELETE FROM {table} WHERE run_id=?', (run_id,))
                    db.execute("UPDATE jobs SET input_digest='',capability_hash=NULL,draft_id=NULL WHERE run_id=?", (run_id,))
                    db.execute("UPDATE runs SET versions='[]',draft_id=NULL,idempotency_key='deleted:'||id,request_hash='' WHERE id=?", (run_id,))
                db.execute("UPDATE agent_tasks SET idempotency_key='deleted:'||id WHERE case_id=?", (case_id,))
                db.execute('DELETE FROM data_permissions WHERE case_id=?', (case_id,))
                db.execute('DELETE FROM imports WHERE case_id=?', (case_id,))
                db.execute('DELETE FROM snapshots WHERE case_id=?', (case_id,))
                core._event(db, None, 'case_content_deleted', case_id=case_id,
                            contract_version=request.contract_version, physical_erasure=False)
        return self.read(owner, case_id)

    @staticmethod
    def _path(session_id):
        if not re.fullmatch(r'[0-9a-f]{32}', session_id):
            raise GateError('SDK_STORAGE_PATH', 503)
        path = SESSION_ROOT / session_id
        # Do not resolve and then delete: even an in-project symlink must not redirect cleanup.
        for part in (path, *path.parents):
            if part.is_symlink():
                raise GateError('SDK_STORAGE_PATH', 503)
            if part == ROOT:
                break
        return path

    def _cleanup(self, session_id):
        with self.core.store.tx() as db:
            row = db.execute('SELECT * FROM sdk_sessions WHERE id=? AND ledger_scope=?',
                             (session_id, self.scope)).fetchone()
            if not row or row['state'] == 'PATH_CONFLICT':
                raise GateError('SDK_STORAGE_OWNERSHIP', 503)
        try:
            path = self._path(session_id)
            if path.exists():
                shutil.rmtree(path)
        except (OSError, GateError):
            with self.core.store.tx() as db:
                db.execute("UPDATE sdk_sessions SET state='CLEANUP_FAILED' WHERE id=?", (session_id,))
                self.core._event(db, row['run_id'], 'sdk_cleanup_failed', session_id=session_id)
            raise GateError('SDK_CLEANUP_FAILED', 503) from None
        with self.core.store.tx() as db:
            db.execute("UPDATE sdk_sessions SET state='REMOVED' WHERE id=?", (session_id,))
            self.core._event(db, row['run_id'], 'sdk_directory_removed', session_id=session_id)

    @contextmanager
    def session(self, owner, run_id, job_id):
        session_id = uuid4().hex
        path = self._path(session_id)
        with self.core.store.tx() as db:
            run = self.core._run(db, run_id, owner)
            self.core._current_job(db, run, job_id)
            self.require_active(db, run['case_id'])
            db.execute('INSERT INTO sdk_sessions VALUES(?,?,?,?,?,?)',
                       (session_id, run['case_id'], run_id, job_id, self.scope, 'ACTIVE'))
            self.core._event(db, run_id, 'sdk_directory_registered', session_id=session_id, job_id=job_id)
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=False)
        except FileExistsError:
            with self.core.store.tx() as db:
                db.execute("UPDATE sdk_sessions SET state='PATH_CONFLICT' WHERE id=?", (session_id,))
                self.core._event(db, run_id, 'sdk_directory_conflict', session_id=session_id)
            raise GateError('SDK_STORAGE_OWNERSHIP', 503) from None
        except OSError:
            with self.core.store.tx() as db:
                db.execute("UPDATE sdk_sessions SET state='CLEANUP_FAILED' WHERE id=?", (session_id,))
                self.core._event(db, run_id, 'sdk_directory_creation_failed', session_id=session_id)
            raise GateError('SDK_STORAGE_FAILED', 503) from None
        try:
            yield session_id
        finally:
            # WorkerRunner returns only after reaping its process group, including timeout/cancel.
            self._cleanup(session_id)

    def recover(self):
        """Call only while holding the exclusive service lock, after interrupted-job fencing."""
        with self.core.store.tx() as db:
            rows = db.execute("SELECT id FROM sdk_sessions WHERE ledger_scope=? AND state!='REMOVED'", (self.scope,)).fetchall()
        failed = 0
        for row in rows:
            try:
                self._cleanup(row['id'])
            except GateError:
                failed += 1
        if failed:
            raise GateError('SDK_CLEANUP_FAILED', 503)
        return {'removed': len(rows), 'failed': failed}
