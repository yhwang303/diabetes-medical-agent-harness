"""Owner-controlled outbound permissions; no agent can grant its own access."""

import json
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field

from .data_lifecycle import lifecycle_catalog
from .contracts import GateError, StrictModel, canonical, digest

POLICY_VERSION = 'engineering-data-policy-v1'
Purpose = Literal['report', 'medical_review', 'ethics_review', 'prediction', 'rl']
PURPOSES = ('report', 'medical_review', 'ethics_review', 'prediction', 'rl')


class DataPermissionRequest(StrictModel):
    snapshot_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    purpose: Purpose
    allowed: bool
    policy_version: Literal['engineering-data-policy-v1']


def data_policy_catalog():
    return {'version': POLICY_VERSION, 'real_personal_data_enabled': False,
        'provider': 'deepseek_official', 'model': 'deepseek-flash',
        'permission_scope': ['case', 'snapshot', 'purpose', 'policy_hash', 'service_instance'],
        'data_scope': 'synthetic_engineering_only', 'default': 'deny', 'restart': 'new_permission_required',
        'local_core': ['snapshot', 'import_filename_hash', 'model_artifacts', 'drafts', 'reviews',
                       'releases', 'permission_records', 'audit_metadata'],
        'outbound': {
            'rl': ['fixed_prompts', 'tool_schemas', 'job_status_and_error_codes', 'job_id',
                   'candidate_artifact_reference_hash_origin_producer_version', 'model_generated_conversation', 'sdk_protocol_metadata'],
            'prediction': ['fixed_prompts', 'tool_schemas', 'job_status_and_error_codes', 'job_id',
                           'artifact_reference_hash_origin_producer_version', 'model_generated_conversation', 'sdk_protocol_metadata'],
            'report': ['fixed_prompts', 'tool_schemas', 'template_version', 'evidence_hash',
                       'controlled_task_status', 'section_proposal', 'model_generated_conversation', 'sdk_protocol_metadata'],
            'medical_review': ['fixed_prompts', 'tool_schemas', 'bound_engineering_report',
                               'report_evidence_hashes', 'run_case_snapshot_draft_revision_config_binding',
                               'structured_review', 'model_generated_conversation', 'sdk_protocol_metadata'],
            'ethics_review': ['fixed_prompts', 'tool_schemas', 'bound_engineering_report',
                              'report_evidence_hashes', 'run_case_snapshot_draft_revision_config_binding',
                              'structured_review', 'model_generated_conversation', 'sdk_protocol_metadata']},
        'never_provided_to_llm': ['raw_snapshot', 'import_filename', 'source_locator', 'other_case_records',
                                 'provider_key', 'permission_management'],
        'observation': {'enabled': False, 'raw_upload': False},
        'logs': 'controlled_metadata_only_no_raw_input_or_model_prose',
        'retention': lifecycle_catalog()}


def job_purpose(step):
    if step in ('policy', 'rl_agent'): return 'rl'
    if step in ('prediction', 'prediction_agent'): return 'prediction'
    return 'report' if step == 'draft' else step.removeprefix('review_') + '_review'


class DataPermissions:
    def __init__(self, core):
        self.core = core
        self.instance = uuid4().hex

    def _case(self, db, owner, case_id):
        case = db.execute('SELECT * FROM cases WHERE id=? AND owner=?', (case_id, owner)).fetchone()
        if not case:
            raise GateError('NOT_FOUND', 404)
        self.core.data_lifecycle.require_active(db, case_id)
        return case

    def _require(self, db, owner, case_id, snapshot_id, purpose):
        case = self._case(db, owner, case_id)
        if case['snapshot_id'] != snapshot_id:
            raise GateError('SNAPSHOT_CHANGED')
        if self.core._snapshot(db, snapshot_id)['source'] != 'synthetic':
            raise GateError('DATA_SCOPE_NOT_SUPPORTED', 403)
        if purpose not in PURPOSES:
            raise GateError('DATA_PURPOSE_FORBIDDEN', 403)
        row = db.execute('SELECT * FROM data_permissions WHERE case_id=? AND purpose=?',
                         (case_id, purpose)).fetchone()
        if (not row or not row['allowed'] or row['snapshot_id'] != snapshot_id
                or row['instance'] != self.instance or row['policy_hash'] != digest(data_policy_catalog())):
            raise GateError('DATA_PERMISSION_REQUIRED', 403)
        return {key: row[key] for key in ('case_id', 'snapshot_id', 'purpose', 'revision', 'instance', 'policy_hash')}

    def read(self, owner, case_id):
        with self.core.store.tx() as db:
            case = self._case(db, owner, case_id)
            permissions = {}
            for purpose in PURPOSES:
                try:
                    self._require(db, owner, case_id, case['snapshot_id'], purpose)
                    permissions[purpose] = True
                except GateError as exc:
                    if exc.code not in ('DATA_PERMISSION_REQUIRED', 'DATA_SCOPE_NOT_SUPPORTED'):
                        raise
                    permissions[purpose] = False
            return {'case_id': case_id, 'snapshot_id': case['snapshot_id'],
                    'policy_version': POLICY_VERSION, 'permissions': permissions}

    def update(self, owner, case_id, raw):
        request = DataPermissionRequest.model_validate(raw)
        with self.core.store.tx() as db:
            case = self._case(db, owner, case_id)
            if case['snapshot_id'] != request.snapshot_id:
                raise GateError('SNAPSHOT_CHANGED')
            if request.allowed and self.core._snapshot(db, request.snapshot_id)['source'] != 'synthetic':
                raise GateError('DATA_SCOPE_NOT_SUPPORTED', 403)
            old = db.execute('SELECT * FROM data_permissions WHERE case_id=? AND purpose=?',
                             (case_id, request.purpose)).fetchone()
            policy_hash = digest(data_policy_catalog())
            same = old and all(old[key] == value for key, value in {
                'snapshot_id': request.snapshot_id, 'allowed': request.allowed,
                'policy_hash': policy_hash, 'instance': self.instance}.items())
            if not same:
                revision = old['revision'] + 1 if old else 1
                db.execute('INSERT OR REPLACE INTO data_permissions VALUES(?,?,?,?,?,?,?)',
                           (case_id, request.purpose, request.snapshot_id, int(request.allowed), revision,
                            policy_hash, self.instance))
                self.core._event(db, None, 'data_permission_changed', case_id=case_id,
                    snapshot_id=request.snapshot_id, purpose=request.purpose, allowed=request.allowed,
                    revision=revision, policy_version=POLICY_VERSION, policy_hash=policy_hash)
        return self.read(owner, case_id)

    def authorize_run(self, owner, run_id, purpose):
        with self.core.store.tx() as db:
            run = self.core._run(db, run_id, owner)
            self.core._live(db, run)
            return self._require(db, owner, run['case_id'], run['snapshot_id'], purpose)

    def bind_job(self, owner, run_id, job_id, permission):
        with self.core.store.tx() as db:
            run = self.core._run(db, run_id, owner)
            job = self.core._current_job(db, run, job_id)
            expected_purpose = job_purpose(job['step'])
            current = self._require(db, owner, run['case_id'], run['snapshot_id'], expected_purpose)
            if permission != current:
                raise GateError('DATA_PERMISSION_CHANGED', 403)
            db.execute('INSERT INTO job_data_permissions VALUES(?,?,?)',
                       (job_id, canonical(current), digest(current)))
            self.core._event(db, run_id, 'data_permission_bound', job_id=job_id,
                purpose=current['purpose'], revision=current['revision'], policy_hash=current['policy_hash'])

    def validate_job(self, db, run, job_id):
        row = db.execute('SELECT * FROM job_data_permissions WHERE job_id=?', (job_id,)).fetchone()
        if row:
            saved = json.loads(row['body'])
            if row['digest'] != digest(saved):
                raise GateError('DATA_PERMISSION_INTEGRITY', 403)
            job = db.execute('SELECT step FROM jobs WHERE id=?', (job_id,)).fetchone()
            purpose = job_purpose(job['step'])
            current = self._require(db, run['owner'], run['case_id'], run['snapshot_id'], purpose)
            if saved != current:
                raise GateError('DATA_PERMISSION_CHANGED', 403)
