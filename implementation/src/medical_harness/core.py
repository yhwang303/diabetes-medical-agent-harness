"""Trusted orchestration, artifact ledger and sole release authority.

Only the API facade is available to agents. Python imports and DB access belong
to the trusted service, not to an adversarial in-process plugin environment.
"""

import json
from fractions import Fraction
import hashlib
import secrets
import time
from uuid import uuid4

from pydantic import ValidationError

from .adapters import REAL_EXECUTORS, basal_fixture_registry, legacy_basal_fixture_registry, fixture_registry, policy_scenario_registry
from .basal_inputs import input_block_reason, parse_snapshot, require_execution_profile
from .basal_actions import (PROFILE_VERSION, LEGACY_PROFILE_VERSION, CONVERSION_VERSION, BasalPredictionOutput,
                            BasalPredictionRequest, BasalPolicyRequest, check_fixture_input,
                            check_output_binding, fixture_profile, legacy_fixture_profile,
                            policy_contract, artifact_contract, rate_per_minute)
from .contracts import (ArtifactEnvelope, CaseEvidence, GateError, ImportRequest, PolicyOutput, PolicyRequest, PredictionOutput, PredictionRequest, Proposal,
                        ReviewOutput, canonical, digest, strict_json)
from .render import KB_VERSION, RULE_VERSION, TEMPLATE_VERSION, TECHNICAL_NOTICE, render
from .store import Store
from .data_lifecycle import DataLifecycle
from .data_policy import DataPermissions
from .workers import WorkerRunner
from .sdk_worker import REVIEW_EXECUTORS, review_configuration


def uid() -> str:
    return uuid4().hex


class Core:
    def __init__(self, store: Store, *, enable_fixtures=False, clock=time.time,
                 executor_timeout=5.0, run_ttl=3600.0, executor_runner=None, review_agent=None,
                 enable_basal_fixtures=False):
        self.store = store
        self.data_permissions = DataPermissions(self)
        self.data_lifecycle = DataLifecycle(self)
        self.enable_fixtures = enable_fixtures
        self.enable_basal_fixtures = enable_basal_fixtures
        self.clock = clock
        self.executor_timeout = executor_timeout
        self.run_ttl = run_ttl
        self._fixtures = fixture_registry()
        self._basal_fixtures = basal_fixture_registry()
        self._legacy_basal_fixtures = legacy_basal_fixture_registry()
        self._policy_scenarios = policy_scenario_registry(executor_timeout)
        self._real = dict(REAL_EXECUTORS)
        self._runner = executor_runner if executor_runner is not None else WorkerRunner()
        self._review_agent = review_agent
        with store.tx() as db:
            versions = [RULE_VERSION, TEMPLATE_VERSION, KB_VERSION]
            versions += [x.version for x in self._fixtures.values()]
            versions += [x.version for x in self._policy_scenarios.values()]
            if enable_basal_fixtures:
                profile = fixture_profile()
                versions += [PROFILE_VERSION, profile['rule_version'], profile['template_version']]
                versions += [x.version for x in self._basal_fixtures.values()]
                legacy = legacy_fixture_profile()
                versions += [LEGACY_PROFILE_VERSION, legacy['rule_version'], legacy['template_version'], CONVERSION_VERSION]
                versions += [x.version for x in self._legacy_basal_fixtures.values()]
            if review_agent is not None:
                versions += [x.version for x in REVIEW_EXECUTORS.values()]
            for version in versions:
                db.execute("INSERT OR IGNORE INTO versions(name) VALUES(?)", (version,))

    def close(self):
        self._runner.close()
        if self._review_agent is not None:
            self._review_agent.close()

    def _event(self, db, run_id, kind, **metadata):
        event_id = uid()
        # Only controlled metadata, never raw model prose, credentials or patient input.
        db.execute("INSERT INTO events(id,run_id,kind,body,time) VALUES(?,?,?,?,?)",
                   (event_id, run_id, kind, canonical(metadata), self.clock()))
        db.execute("INSERT INTO outbox(event_id) VALUES(?)", (event_id,))

    def record_denial(self, owner, run_id, code):
        with self.store.tx() as db:
            # Do not allow guessed IDs to contaminate someone else's audit history.
            if run_id:
                self._run(db, run_id, owner)
            self._event(db, run_id, "request_denied", code=code)

    def create_case(self, owner: str, raw: dict) -> dict:
        body = parse_snapshot(raw).model_dump()
        case_id, snapshot_id = uid(), uid()
        with self.store.tx() as db:
            db.execute("INSERT INTO cases VALUES(?,?,?)", (case_id, owner, snapshot_id))
            db.execute("INSERT INTO snapshots VALUES(?,?,?,?)",
                       (snapshot_id, case_id, canonical(body), digest(body)))
            self._event(db, None, "snapshot_frozen", case_id=case_id, snapshot_id=snapshot_id)
        return {"case_id": case_id, "snapshot_id": snapshot_id}

    def update_case(self, owner: str, case_id: str, raw: dict) -> dict:
        body = parse_snapshot(raw).model_dump()
        snapshot_id = uid()
        with self.store.tx() as db:
            case = db.execute("SELECT * FROM cases WHERE id=? AND owner=?", (case_id, owner)).fetchone()
            if not case:
                raise GateError("NOT_FOUND", 404)
            self.data_lifecycle.require_active(db, case_id)
            db.execute("INSERT INTO snapshots VALUES(?,?,?,?)",
                       (snapshot_id, case_id, canonical(body), digest(body)))
            db.execute("UPDATE cases SET snapshot_id=? WHERE id=?", (snapshot_id, case_id))
            for run in db.execute("SELECT id FROM runs WHERE case_id=?", (case_id,)).fetchall():
                self._invalidate(db, run["id"], "SNAPSHOT_CHANGED")
            self._event(db, None, "snapshot_replaced", case_id=case_id, snapshot_id=snapshot_id)
        return {"case_id": case_id, "snapshot_id": snapshot_id}

    def import_case(self, owner, raw):
        incoming = ImportRequest.model_validate(raw)
        body = parse_snapshot(strict_json(incoming.content)).model_dump()
        try:
            canonical(body).encode("utf-8")
        except UnicodeError as exc:
            raise GateError("INVALID_JSON", 422) from exc
        file_hash = hashlib.sha256(incoming.content.encode("utf-8")).hexdigest()
        with self.store.tx() as db:
            # Retrying exactly the same file for this owner must not duplicate a case.
            old = db.execute("""SELECT c.id,c.snapshot_id FROM imports i JOIN cases c ON c.id=i.case_id
                WHERE c.owner=? AND i.file_hash=? AND c.snapshot_id=i.snapshot_id""", (owner, file_hash)).fetchone()
            if old:
                return {"case_id": old["id"], "snapshot_id": old["snapshot_id"], "duplicate": True}
            case_id, snapshot_id = uid(), uid()
            db.execute("INSERT INTO cases VALUES(?,?,?)", (case_id, owner, snapshot_id))
            db.execute("INSERT INTO snapshots VALUES(?,?,?,?)", (snapshot_id, case_id, canonical(body), digest(body)))
            db.execute("INSERT INTO imports VALUES(?,?,?,?,?,?)", (case_id, snapshot_id, incoming.file_name,
                       file_hash, len(incoming.content.encode("utf-8")), self.clock()))
            self._event(db, None, "file_imported", case_id=case_id, snapshot_id=snapshot_id,
                        file_hash=file_hash, snapshot_hash=digest(body))
        return {"case_id": case_id, "snapshot_id": snapshot_id, "duplicate": False}

    def list_cases(self, owner):
        with self.store.tx() as db:
            rows = db.execute("""SELECT c.id AS case_id,c.snapshot_id,i.file_name,i.imported_at
                FROM cases c LEFT JOIN imports i ON i.case_id=c.id AND i.snapshot_id=c.snapshot_id
                WHERE c.owner=? AND NOT EXISTS (SELECT 1 FROM case_deletions d WHERE d.case_id=c.id) ORDER BY COALESCE(i.imported_at,0) DESC,c.id DESC LIMIT 100""", (owner,)).fetchall()
            return [dict(row) for row in rows]

    def input_timeline(self, owner, case_id):
        # This endpoint exposes the owner's frozen input only, never generated candidates/drafts.
        with self.store.tx() as db:
            case = db.execute("SELECT * FROM cases WHERE id=? AND owner=?", (case_id, owner)).fetchone()
            if not case:
                raise GateError("NOT_FOUND", 404)
            self.data_lifecycle.require_active(db, case_id)
            snapshot = self._snapshot(db, case["snapshot_id"])
            imported = db.execute("SELECT * FROM imports WHERE case_id=? AND snapshot_id=?",
                                  (case_id, case["snapshot_id"])).fetchone()
            return {"case_id": case_id, "snapshot_id": case["snapshot_id"], "snapshot": snapshot,
                    "snapshot_hash": digest(snapshot), "import": dict(imported) if imported else None,
                    "content_type": "input_observations", "clinical_interpretation": False}

    def start_run(self, owner: str, case_id: str, mode: str, idempotency_key: str,
                  expected_snapshot_id: str | None = None, fixture_scenario: str | None = None) -> dict:
        if mode not in {"eval", "research"}:
            raise GateError("INVALID_MODE", 422)
        if mode == "eval" and not self.enable_fixtures:
            raise GateError("FIXTURES_DISABLED", 403)
        if fixture_scenario is not None:
            if mode != "eval":
                raise GateError("FIXTURE_FORBIDDEN", 403)
            if fixture_scenario not in self._policy_scenarios:
                raise GateError("INVALID_FIXTURE_SCENARIO", 422)
        if not 1 <= len(idempotency_key) <= 128:
            raise GateError("INVALID_IDEMPOTENCY_KEY", 422)
        with self.store.tx() as db:
            case = db.execute("SELECT * FROM cases WHERE id=? AND owner=?", (case_id, owner)).fetchone()
            if not case:
                raise GateError("NOT_FOUND", 404)
            self.data_lifecycle.require_active(db, case_id)
            request_identity = {"case": case_id, "snapshot": case["snapshot_id"], "mode": mode}
            if fixture_scenario is not None:
                request_identity["fixture_scenario"] = fixture_scenario
            request_hash = digest(request_identity)
            if expected_snapshot_id is not None and case["snapshot_id"] != expected_snapshot_id:
                raise GateError("SNAPSHOT_CHANGED")
            existing = db.execute("SELECT * FROM runs WHERE owner=? AND idempotency_key=?",
                                  (owner, idempotency_key)).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise GateError("IDEMPOTENCY_CONFLICT")
                return self._status(existing)
            snapshot = self._snapshot(db, case["snapshot_id"])
            profile = self._input_profile(snapshot, mode, fixture_scenario)
            if mode == "eval" and snapshot["source"] != "synthetic":
                raise GateError("FIXTURE_REQUIRES_SYNTHETIC_CASE")
            versions = [RULE_VERSION, TEMPLATE_VERSION, KB_VERSION]
            if profile:
                versions = [PROFILE_VERSION, profile['rule_version'], profile['template_version'], KB_VERSION, CONVERSION_VERSION]
                versions += [x.version for x in self._basal_fixtures.values()]
            if mode == "eval":
                versions += [x.version for name, x in self._fixtures.items()
                             if (self._review_agent is None or name not in REVIEW_EXECUTORS)
                             and (not profile or name in REVIEW_EXECUTORS)]
            if self._review_agent is not None:
                versions += [x.version for x in REVIEW_EXECUTORS.values()]
            if fixture_scenario is not None:
                versions.append(self._policy_scenarios[fixture_scenario].version)
            self._check_versions(db, versions)
            run_id = uid()
            expires = self.clock() + (min(self.run_ttl, profile['max_wall_seconds']) if profile else self.run_ttl)
            if profile:
                profile = dict(profile, binding={'run_id': run_id, 'case_id': case_id,
                    'snapshot_id': case['snapshot_id'], 'snapshot_hash': digest(snapshot),
                    'decision_time': snapshot['decision_time'], 'expires_at': expires})
            db.execute("""INSERT INTO runs(id,case_id,snapshot_id,owner,mode,state,expires,versions,
                       idempotency_key,request_hash,fixture_scenario) VALUES(?,?,?,?,?,'SNAPSHOT_FROZEN',?,?,?,?,?)""",
                       (run_id, case_id, case["snapshot_id"], owner, mode, expires,
                        canonical(versions), idempotency_key, request_hash, fixture_scenario))
            if profile:
                db.execute('INSERT INTO run_profiles VALUES(?,?,?)', (run_id, canonical(profile), digest(profile)))
                self._event(db, run_id, 'input_profile_frozen', version=PROFILE_VERSION, profile_hash=digest(profile))
            if self._review_agent is not None:
                for role in REVIEW_EXECUTORS:
                    config = review_configuration(role)
                    db.execute('INSERT INTO review_configs VALUES(?,?,?,?)',
                               (run_id, role, canonical(config), digest(config)))
            self._event(db, run_id, "run_created", mode=mode, snapshot_id=case["snapshot_id"], fixture_scenario=fixture_scenario)
            return self._status(self._run(db, run_id, owner))

    @staticmethod
    def _status(run):
        return {key: run[key] for key in ("id", "case_id", "snapshot_id", "mode", "state", "reason", "revision", "fixture_scenario")}

    def _run(self, db, run_id, owner):
        run = db.execute("SELECT * FROM runs WHERE id=? AND owner=?", (run_id, owner)).fetchone()
        if not run:
            raise GateError("NOT_FOUND", 404)
        return run

    def _snapshot(self, db, snapshot_id):
        row = db.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            raise GateError("SNAPSHOT_MISSING")
        body = json.loads(row["body"])
        if digest(body) != row["digest"]:
            raise GateError("SNAPSHOT_INTEGRITY")
        parse_snapshot(body)
        return body

    def _check_versions(self, db, versions):
        for name in versions:
            row = db.execute("SELECT revoked FROM versions WHERE name=?", (name,)).fetchone()
            if not row or row["revoked"]:
                raise GateError("VERSION_REVOKED")

    def _input_profile(self, snapshot, mode, scenario=None):
        if 'contract_version' not in snapshot:
            return None
        if not self.enable_basal_fixtures:
            require_execution_profile(snapshot)
        check_fixture_input(snapshot, mode)
        if scenario is not None:
            raise GateError('FIXTURE_SCENARIO_PROFILE_MISMATCH')
        return fixture_profile()

    def _profile(self, db, run):
        snapshot = self._snapshot(db, run['snapshot_id'])
        expected = self._input_profile(snapshot, run['mode'], run['fixture_scenario'])
        row = db.execute('SELECT * FROM run_profiles WHERE run_id=?', (run['id'],)).fetchone()
        if expected is None:
            if row or {PROFILE_VERSION, LEGACY_PROFILE_VERSION} & set(json.loads(run['versions'])):
                raise GateError('INPUT_PROFILE_BINDING')
            return None
        if not row:
            raise GateError('INPUT_PROFILE_MISSING')
        run_versions = set(json.loads(run['versions']))
        if {PROFILE_VERSION, LEGACY_PROFILE_VERSION} <= run_versions:
            raise GateError('INPUT_PROFILE_BINDING')
        if LEGACY_PROFILE_VERSION in run_versions:
            expected = legacy_fixture_profile()
        expected = dict(expected, binding={'run_id': run['id'], 'case_id': run['case_id'],
            'snapshot_id': run['snapshot_id'], 'snapshot_hash': digest(snapshot),
            'decision_time': snapshot['decision_time'], 'expires_at': run['expires']})
        saved = strict_json(row['body'])
        if digest(saved) != row['digest']:
            raise GateError('INPUT_PROFILE_INTEGRITY')
        if saved != expected:
            raise GateError('INPUT_PROFILE_CHANGED')
        required = {saved['version'], saved['rule_version'], saved['template_version'],
                    saved['prediction']['version'], saved['policy']['version']}
        if saved['version'] == PROFILE_VERSION:
            required.add(CONVERSION_VERSION)
        if not required <= run_versions:
            raise GateError('INPUT_PROFILE_BINDING')
        for step in ('prediction', 'policy'):
            registry = self._legacy_basal_fixtures if saved['version'] == LEGACY_PROFILE_VERSION else self._basal_fixtures
            executor = registry[step]
            if (executor.identity, executor.version, executor.origin) != (saved[step]['identity'], saved[step]['version'], 'fixture'):
                raise GateError('EXECUTOR_PROFILE_MISMATCH')
        return saved

    def _live(self, db, run):
        self.data_lifecycle.require_active(db, run["case_id"])
        if run["state"] in {"CANCELLED", "INVALIDATED", "BLOCKED", "FAILED"}:
            raise GateError(run["reason"] or run["state"])
        if self.clock() >= run["expires"]:
            raise GateError("RUN_EXPIRED")
        case = db.execute("SELECT snapshot_id FROM cases WHERE id=?", (run["case_id"],)).fetchone()
        if case["snapshot_id"] != run["snapshot_id"]:
            raise GateError("SNAPSHOT_CHANGED")
        self._profile(db, run)
        self._check_versions(db, json.loads(run["versions"]))
        for role in REVIEW_EXECUTORS:
            if self._uses_sdk_review(run, role):
                self._review_configuration(db, run, role)
        if run["mode"] == "eval" and not self.enable_fixtures:
            raise GateError("FIXTURES_DISABLED", 403)

    def status(self, owner, run_id):
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            result = self._status(run)
            try:
                self._live(db, run)
                result["currently_valid"] = True
            except GateError as exc:
                result.update(currently_valid=False, validity_reason=exc.code)
            # No raw candidate, draft or rejected result in progress/status responses.
            return result

    def case_evidence(self, owner, case_id):
        """Read-only gate projection. No payloads, draft text, verdict text, or state mutation."""
        with self.store.tx() as db:
            case = db.execute("SELECT * FROM cases WHERE id=? AND owner=?", (case_id, owner)).fetchone()
            if not case:
                raise GateError("NOT_FOUND", 404)
            self.data_lifecycle.require_active(db, case_id)
            snapshot = self._snapshot(db, case["snapshot_id"])
            missing = sum(snapshot["missing_mask"])
            input_reason = input_block_reason(snapshot)
            if self.enable_basal_fixtures and 'contract_version' in snapshot:
                try:
                    self._input_profile(snapshot, 'eval')
                    input_reason = None
                except GateError as exc:
                    input_reason = exc.code
            run = db.execute("SELECT * FROM runs WHERE case_id=? AND owner=? ORDER BY rowid DESC LIMIT 1",
                             (case_id, owner)).fetchone()
            cards = [{"kind": "input", "status": "blocked" if input_reason else "verified",
                      "reason": input_reason,
                      "reference_id": case["snapshot_id"], "digest": digest(snapshot)}]
            invalid = None
            if run:
                try:
                    self._live(db, run)
                except GateError as exc:
                    invalid = exc.code
            for kind in ("prediction", "policy"):
                card = {"kind": kind, "status": "waiting", "reason": "RUN_NOT_STARTED"}
                if invalid:
                    card.update(status="blocked", reason=invalid)
                elif run:
                    try:
                        body, artifact_hash = self._artifact(db, run, kind)
                        card.update(status="verified", reason=None, reference_id=body["id"],
                                    digest=artifact_hash, producer=body["producer"], version=body["version"], origin=body["origin"])
                    except GateError as exc:
                        reason = exc.code
                        if reason == "MISSING_" + kind.upper():
                            # Show actual current attempt failure; policy must not inherit a prediction failure.
                            job = db.execute("SELECT status FROM jobs WHERE run_id=? AND step=? ORDER BY attempt DESC LIMIT 1",
                                             (run["id"], kind)).fetchone()
                            if job and job["status"] == "FAILED":
                                reason = run["reason"] or "MODEL_RESULT_MISSING"
                        card.update(status="blocked" if reason != "MISSING_" + kind.upper() else "waiting", reason=reason)
                elif kind not in self._real:
                    card.update(status="blocked", reason="MODEL_NOT_CONFIGURED")
                if input_reason and card["status"] != "verified":
                    card.update(status="blocked", reason=input_reason)
                cards.append(card)
            safety = {"kind": "safety", "status": "waiting", "reason": "MODEL_EVIDENCE_REQUIRED"}
            if invalid:
                safety.update(status="blocked", reason=invalid)
            elif run and all(c["status"] == "verified" for c in cards):
                try:
                    _, _, manifest = self._evidence(db, run)
                    safety.update(status="verified", reason=None, digest=digest(manifest), version=manifest['rule_version'])
                except GateError as exc:
                    safety.update(status="blocked", reason=exc.code)
            cards.append(safety)
            reviews = {"kind": "reviews", "status": "waiting", "reason": "REVIEW_REQUIRED"}
            release = {"kind": "release", "status": "blocked", "reason": "REVIEW_REQUIRED"}
            if invalid:
                reviews.update(status="blocked", reason=invalid)
                release["reason"] = invalid
            elif run and safety["status"] == "verified":
                try:
                    draft, _, _ = self._release_checks(db, run)
                    reviews.update(status="verified", reason=None, reference_id=draft["id"], digest=draft["digest"])
                    release.update(status="verified", reason=None)
                except GateError as exc:
                    reason = run["reason"] if run["state"] == "REVIEW_BLOCKED" else exc.code
                    reviews.update(status="blocked" if run["state"] == "REVIEW_BLOCKED" or exc.code != "REVIEW_REQUIRED" else "waiting",
                                   reason=reason or exc.code)
                    release["reason"] = reason or exc.code
            elif safety["status"] != "verified":
                release["reason"] = safety["reason"]
            cards.extend([reviews, release])
            return CaseEvidence.model_validate({"case_id": case_id, "snapshot_id": case["snapshot_id"],
                "source": snapshot["source"], "missing_count": missing, "run_id": run["id"] if run else None,
                "run_state": run["state"] if run else "NOT_STARTED", "mode": run["mode"] if run else None,
                "fixture_scenario": run["fixture_scenario"] if run else None,
                "checked_at": self.clock(), "release_allowed": all(c["status"] == "verified" for c in cards),
                "cards": cards}).model_dump()

    def _invalidate(self, db, run_id, reason):
        db.execute("UPDATE runs SET state='INVALIDATED',reason=? WHERE id=?", (reason, run_id))
        db.execute("UPDATE jobs SET status='FENCED' WHERE run_id=? AND status='RUNNING'", (run_id,))
        self._event(db, run_id, "run_invalidated", reason=reason)

    def cancel(self, owner, run_id):
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            if run["state"] != "CANCELLED":
                db.execute("UPDATE runs SET state='CANCELLED',reason='CANCELLED' WHERE id=?", (run_id,))
                db.execute("UPDATE jobs SET status='FENCED' WHERE run_id=? AND status='RUNNING'", (run_id,))
                self._event(db, run_id, "run_cancelled")
        return self.status(owner, run_id)

    def revoke_version(self, version):
        # Trusted control-plane method; never exposed as an agent tool.
        with self.store.tx() as db:
            if not db.execute("SELECT 1 FROM versions WHERE name=?", (version,)).fetchone():
                raise GateError("NOT_FOUND", 404)
            db.execute("UPDATE versions SET revoked=1 WHERE name=?", (version,))
            for run in db.execute("SELECT id,versions FROM runs").fetchall():
                if version in json.loads(run["versions"]):
                    self._invalidate(db, run["id"], "VERSION_REVOKED")

    def _new_job(self, db, run, step, input_digest, *, capability_hash=None, draft_id=None):
        if db.execute("SELECT 1 FROM jobs WHERE run_id=? AND step=? AND status='RUNNING'",
                      (run["id"], step)).fetchone():
            raise GateError("JOB_IN_PROGRESS")
        attempt = db.execute("SELECT COALESCE(MAX(attempt),0)+1 FROM jobs WHERE run_id=? AND step=?",
                             (run["id"], step)).fetchone()[0]
        if attempt > 3:
            raise GateError("ATTEMPT_BUDGET_EXHAUSTED")
        job_id = uid()
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                   (job_id, run["id"], step, attempt, "RUNNING", self.clock() + self.executor_timeout,
                    input_digest, capability_hash, draft_id))
        self._event(db, run["id"], "job_started", job_id=job_id, step=step, attempt=attempt)
        return job_id

    def _current_job(self, db, run, job_id):
        self._live(db, run)
        job = db.execute("SELECT * FROM jobs WHERE id=? AND run_id=?", (job_id, run["id"])).fetchone()
        if not job or job["status"] != "RUNNING":
            raise GateError("STALE_ATTEMPT")
        if self.clock() >= job["deadline"]:
            raise GateError("JOB_EXPIRED")
        self.data_permissions.validate_job(db, run, job_id)
        self._check_model_controller(db, run, job, active=True)
        return job

    def _check_model_controller(self, db, run, job, *, active=False, completed=False):
        dependency = db.execute('SELECT parent_job_id FROM job_dependencies WHERE child_job_id=?', (job['id'],)).fetchone()
        if job['step'] not in ('prediction', 'policy'):
            if dependency: raise GateError('SDK_JOB_BINDING')
            return
        from .prediction_agent import binding as prediction_binding
        from .rl_agent import binding as rl_binding
        prefix, controller_step, binding = (('PREDICTION', 'prediction_agent', prediction_binding)
            if job['step'] == 'prediction' else ('RL', 'rl_agent', rl_binding))
        parent = db.execute('SELECT * FROM jobs WHERE run_id=? AND step=?', (run['id'], controller_step)).fetchone()
        if not parent and not dependency:
            # Legacy SDK-independent fixtures have no controller or outbound grant.
            if db.execute('SELECT 1 FROM job_data_permissions WHERE job_id=?', (job['id'],)).fetchone():
                raise GateError(prefix + '_JOB_BINDING')
            return
        if not parent or not dependency or dependency['parent_job_id'] != parent['id']:
            raise GateError(prefix + '_JOB_BINDING')
        self._live(db, run)
        for required_id in (job['id'], parent['id']):
            if not db.execute('SELECT 1 FROM job_data_permissions WHERE job_id=?', (required_id,)).fetchone():
                raise GateError('DATA_PERMISSION_INTEGRITY', 403)
            self.data_permissions.validate_job(db, run, required_id)
        if active or parent['status'] == 'RUNNING':
            if completed: raise GateError(prefix + '_AGENT_INCOMPLETE')
            self._current_job(db, run, parent['id'])
        elif parent['status'] != 'ACCEPTED':
            raise GateError(prefix + '_AGENT_INCOMPLETE')
        if parent['input_digest'] != binding(self, db, run):
            raise GateError(prefix + '_AGENT_BINDING')

    def _invoke(self, executor, request, owner, run_id, job_id):
        def validate():
            with self.store.tx() as db:
                run = self._run(db, run_id, owner)
                try:
                    job = self._current_job(db, run, job_id)
                except GateError as exc:
                    if exc.code == 'JOB_EXPIRED':
                        raise GateError('EXECUTOR_TIMEOUT', 503) from exc
                    raise
                return job['deadline'] - self.clock()

        def notify(kind, **metadata):
            with self.store.tx() as db:
                self._run(db, run_id, owner)
                self._event(db, run_id, kind, job_id=job_id, **metadata)

        return self._runner.run(executor, request, timeout=validate(), validate=validate,
                                job_id=job_id, notify=notify)

    def _job_failure(self, owner, run_id, job_id, code, *, unavailable=False):
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            # Cancellation, replacement or revocation wins over a late failure.
            if job and job["status"] == "RUNNING":
                db.execute("UPDATE jobs SET status='FAILED' WHERE id=?", (job_id,))
                if run["state"] not in {"CANCELLED", "INVALIDATED"}:
                    step = db.execute("SELECT step FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
                    state = "REVIEW_BLOCKED" if step.startswith("review_") else "UNAVAILABLE" if unavailable else "BLOCKED"
                    db.execute("UPDATE runs SET state=?,reason=? WHERE id=?",
                               (state, code, run_id))
                self._event(db, run_id, "job_failed", job_id=job_id, code=code)

    def _executor(self, run, step):
        if step in REVIEW_EXECUTORS and REVIEW_EXECUTORS[step].version in json.loads(run['versions']):
            if self._review_agent is None:
                raise GateError('REVIEWER_NOT_CONFIGURED', 503)
            return REVIEW_EXECUTORS[step]
        registry = self._fixtures if run["mode"] == "eval" else self._real
        executor = registry.get(step)
        if {PROFILE_VERSION, LEGACY_PROFILE_VERSION} & set(json.loads(run['versions'])) and step in self._basal_fixtures:
            if not self.enable_basal_fixtures or run['mode'] != 'eval':
                raise GateError('INPUT_PROFILE_NOT_CONFIGURED')
            registry = self._legacy_basal_fixtures if LEGACY_PROFILE_VERSION in json.loads(run['versions']) else self._basal_fixtures
            executor = registry[step]
        if run["fixture_scenario"] is not None:
            if run["mode"] != "eval":
                raise GateError("FIXTURE_FORBIDDEN")
            scenario_executor = self._policy_scenarios.get(run["fixture_scenario"])
            if scenario_executor is None:
                raise GateError("INVALID_FIXTURE_SCENARIO")
            if step == "policy":
                executor = scenario_executor
        if executor is None:
            raise GateError("MODEL_NOT_CONFIGURED", 503)
        if run["mode"] != "eval" and executor.origin == "fixture":
            raise GateError("FIXTURE_FORBIDDEN")
        if executor.origin != ("fixture" if run["mode"] == "eval" else "model"):
            raise GateError("EXECUTOR_ORIGIN_MISMATCH")
        if executor.version not in json.loads(run["versions"]):
            raise GateError("EXECUTOR_VERSION_MISMATCH")
        return executor

    @staticmethod
    def _uses_sdk_review(run, role):
        versions = json.loads(run['versions'])
        return REVIEW_EXECUTORS[role].version in versions or any(name.startswith('flash-' + role + '-') for name in versions)

    def _review_configuration(self, db, run, role):
        row = db.execute('SELECT * FROM review_configs WHERE run_id=? AND role=?', (run['id'], role)).fetchone()
        if not row:
            # Do not fabricate a historical fingerprint from today's configuration.
            raise GateError('REVIEW_CONFIG_MISSING')
        try:
            config = strict_json(row['body'])
            if digest(config) != row['digest']:
                raise GateError('REVIEW_CONFIG_INTEGRITY')
        except (ValueError, TypeError) as exc:
            raise GateError('REVIEW_CONFIG_INTEGRITY') from exc
        if config != review_configuration(role) or config['version'] not in json.loads(run['versions']):
            raise GateError('REVIEW_CONFIG_CHANGED')
        return row['digest']

    def _review_request(self, db, run, draft, body, role):
        request = {'report': body, 'report_hash': draft['digest'], 'evidence_hash': draft['evidence_hash']}
        if self._uses_sdk_review(run, role):
            request['binding'] = {'run_id': run['id'], 'case_id': run['case_id'], 'snapshot_id': run['snapshot_id'],
                'draft_id': draft['id'], 'revision': draft['revision'], 'role': role,
                'config_hash': self._review_configuration(db, run, role)}
        return request

    def _validate_review_record(self, db, run, draft, body, row):
        try:
            review = strict_json(row['body'])
            real = self._uses_sdk_review(run, row['role'])
            if (real and row['digest'] is None) or (row['digest'] is not None and digest(review) != row['digest']):
                raise GateError('REVIEW_INTEGRITY')
            keys = {'verdict', 'issues', 'report_hash', 'evidence_hash'}
            ReviewOutput.model_validate({key: review[key] for key in keys})
            if set(review) != keys | {'producer', 'version', 'origin', 'role', 'job_id'} | ({'binding'} if real else set()):
                raise GateError('REVIEW_INVALID')
            executor = self._executor(run, row['role'])
            request = self._review_request(db, run, draft, body, row['role'])
            job = db.execute('SELECT * FROM jobs WHERE id=?', (row['job_id'],)).fetchone()
            if (row['run_id'] != run['id'] or row['draft_id'] != draft['id'] or review['verdict'] != row['verdict']
                    or review['report_hash'] != draft['digest'] or review['evidence_hash'] != draft['evidence_hash']
                    or review['producer'] != executor.identity or review['version'] != executor.version
                    or review['origin'] != executor.origin or review['role'] != row['role'] or review['job_id'] != row['job_id']
                    or (real and review['binding'] != request['binding']) or not job or job['status'] != 'ACCEPTED'
                    or job['run_id'] != run['id'] or job['draft_id'] != draft['id']
                    or job['step'] != 'review_' + row['role'] or job['input_digest'] != digest(request)
                    or (real and review['verdict'] != 'pass' and not review['issues'])):
                raise GateError('REVIEW_INVALID')
            return review
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            raise GateError('REVIEW_INVALID') from exc

    def _artifact(self, db, run, kind, *, completed_controller=False):
        profile = self._profile(db, run)
        row = db.execute("SELECT * FROM artifacts WHERE run_id=? AND kind=?", (run["id"], kind)).fetchone()
        if not row:
            raise GateError("MISSING_" + kind.upper())
        body = json.loads(row["body"])
        try:
            (artifact_contract(profile) if profile else ArtifactEnvelope).model_validate(body)
        except ValidationError as exc:
            raise GateError("ARTIFACT_SCHEMA") from exc
        if digest(body) != row["digest"] or body["id"] != row["id"]:
            raise GateError("ARTIFACT_INTEGRITY")
        if any(body[key] != run[key] for key in ("case_id", "snapshot_id")) or body["run_id"] != run["id"]:
            raise GateError("ARTIFACT_BINDING")
        if body["origin"] != ("fixture" if run["mode"] == "eval" else "model"):
            raise GateError("ARTIFACT_ORIGIN")
        job = db.execute("SELECT * FROM jobs WHERE id=?", (row["job_id"],)).fetchone()
        if not job or job["status"] != "ACCEPTED" or job["run_id"] != run["id"] or job["step"] != kind:
            raise GateError("ARTIFACT_JOB")
        if (body["kind"] != kind or body["job_id"] != job["id"] or body["attempt"] != job["attempt"]
                or body["input_digest"] != job["input_digest"]
                or body["payload"]["input_digest"] != job["input_digest"]):
            raise GateError("ARTIFACT_JOB_BINDING")
        self._check_model_controller(db, run, job, completed=completed_controller)
        request = {"snapshot": self._snapshot(db, run["snapshot_id"]),
                   "snapshot_id": run["snapshot_id"], "run_id": run["id"]}
        if profile:
            request['profile_hash'] = digest(profile)
            check_output_binding(request['snapshot'], body['payload'], digest(profile), kind)
        if kind == "policy":
            prediction, parent = self._artifact(db, run, "prediction", completed_controller=True)
            request.update(prediction=prediction["payload"], prediction_hash=parent)
            if body["parent_hash"] != parent or body["payload"]["forecast_parent_hash"] != parent:
                raise GateError("FORECAST_PARENT_MISMATCH")
        if digest(request) != job["input_digest"]:
            raise GateError("ARTIFACT_INPUT_BINDING")
        executor = self._executor(run, kind)
        if body["producer"] != executor.identity or body["version"] != executor.version:
            raise GateError("ARTIFACT_PRODUCER")
        self._check_versions(db, [body["version"]])
        return body, row["digest"]

    def _model_step(self, owner, run_id, step, *, authorization_job_id=None):
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            self._live(db, run)
            if authorization_job_id is not None:
                from .prediction_agent import PredictionTools
                from .rl_agent import RLTools
                if step not in ('prediction', 'policy'): raise GateError('SDK_TOOL_DENIED')
                tools = PredictionTools if step == 'prediction' else RLTools
                tools(self, owner, run_id, authorization_job_id).validate(db)
            else:
                controller = 'prediction_agent' if step == 'prediction' else 'rl_agent'
                if db.execute("SELECT 1 FROM jobs WHERE run_id=? AND step=? AND status='RUNNING'", (run_id, controller)).fetchone():
                    raise GateError('PREDICTION_SESSION_REQUIRED' if step == 'prediction' else 'RL_SESSION_REQUIRED')
            existing = db.execute("SELECT 1 FROM artifacts WHERE run_id=? AND kind=?", (run_id, step)).fetchone()
            if existing:
                self._artifact(db, run, step)
                return
            allowed = {"SNAPSHOT_FROZEN", "UNAVAILABLE"} if step == "prediction" else {"PREDICTION_ACCEPTED", "UNAVAILABLE"}
            if run["state"] not in allowed:
                raise GateError("INVALID_STAGE")
            snapshot = self._snapshot(db, run["snapshot_id"])
            profile = self._profile(db, run)
            if any(snapshot["missing_mask"]):
                db.execute("UPDATE runs SET state='BLOCKED',reason='MISSING_INPUT' WHERE id=?", (run_id,))
                self._event(db, run_id, "eligibility_rejected", code="MISSING_INPUT")
                return
            request = {"snapshot": snapshot, "snapshot_id": run["snapshot_id"], "run_id": run_id}
            if profile:
                request['profile_hash'] = digest(profile)
            if step == "policy":
                prediction, parent_hash = self._artifact(db, run, "prediction", completed_controller=True)
                request.update(prediction=prediction["payload"], prediction_hash=parent_hash)
            request["input_digest"] = digest(request)
            request_model = (BasalPredictionRequest if step == 'prediction' else BasalPolicyRequest) if profile else (PredictionRequest if step == 'prediction' else PolicyRequest)
            request_model.model_validate(request)
            job_id = self._new_job(db, run, step, request["input_digest"])
            if authorization_job_id is not None:
                grant = db.execute('SELECT body,digest FROM job_data_permissions WHERE job_id=?', (authorization_job_id,)).fetchone()
                if not grant: raise GateError('DATA_PERMISSION_REQUIRED', 403)
                db.execute('INSERT INTO job_data_permissions VALUES(?,?,?)', (job_id, grant['body'], grant['digest']))
                db.execute('INSERT INTO job_dependencies VALUES(?,?)', (job_id, authorization_job_id))
                self._event(db, run_id, 'prediction_job_bound' if step == 'prediction' else 'rl_job_bound', job_id=job_id, parent_job_id=authorization_job_id)
            db.execute("UPDATE runs SET state=?,reason=NULL WHERE id=?", (step.upper() + "_RUNNING", run_id))
        try:
            executor = self._executor(run, step)
            raw = self._invoke(executor, request, owner, run_id, job_id)
            model = (BasalPredictionOutput if step == 'prediction' else policy_contract(profile)) if profile else (PredictionOutput if step == 'prediction' else PolicyOutput)
            output = model.model_validate(raw).model_dump()
            if output["input_digest"] != request["input_digest"]:
                raise GateError("INPUT_DIGEST_MISMATCH")
            if profile:
                check_output_binding(snapshot, output, digest(profile), step)
            if step == "policy":
                if output["forecast_parent_hash"] != request["prediction_hash"]:
                    raise GateError("FORECAST_PARENT_MISMATCH")
                if output["status"] != "candidate":
                    raise GateError("POLICY_" + output["status"].upper())
            with self.store.tx() as db:
                current = self._run(db, run_id, owner)
                job = self._current_job(db, current, job_id)
                if job["input_digest"] != request["input_digest"]:
                    raise GateError("JOB_INPUT_MISMATCH")
                if step == "policy":
                    _, current_parent = self._artifact(db, current, "prediction", completed_controller=True)
                    if current_parent != output["forecast_parent_hash"]:
                        raise GateError("FORECAST_PARENT_MISMATCH")
                artifact_id = uid()
                envelope = {"id": artifact_id, "schema_version": 1, "kind": step, "run_id": run_id,
                            "case_id": current["case_id"], "snapshot_id": current["snapshot_id"],
                            "job_id": job_id, "attempt": job["attempt"], "producer": executor.identity,
                            "version": executor.version, "origin": executor.origin,
                            "input_digest": request["input_digest"],
                            "parent_hash": request.get("prediction_hash"), "payload": output,
                            "accepted_at": self.clock()}
                (artifact_contract(profile) if profile else ArtifactEnvelope).model_validate(envelope)
                db.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?)",
                           (artifact_id, run_id, step, canonical(envelope), digest(envelope), job_id))
                db.execute("UPDATE jobs SET status='ACCEPTED' WHERE id=?", (job_id,))
                db.execute("UPDATE runs SET state=? WHERE id=?", (step.upper() + "_ACCEPTED", run_id))
                self._event(db, run_id, "artifact_accepted", artifact_id=artifact_id, artifact_kind=step,
                            artifact_hash=digest(envelope), job_id=job_id)
        except Exception as exc:
            code = exc.code if isinstance(exc, GateError) else "INVALID_MODEL_OUTPUT" if isinstance(exc, ValidationError) else "EXECUTOR_ERROR"
            self._job_failure(owner, run_id, job_id, code,
                              unavailable=code.startswith('WORKER_') or code in {"MODEL_NOT_CONFIGURED", "EXECUTOR_TIMEOUT", "EXECUTOR_ERROR", "EXECUTOR_BUSY", "CORE_SHUTTING_DOWN"})
            if isinstance(exc, GateError):
                raise
            raise GateError(code) from exc

    def execute_prediction(self, owner, run_id):
        self._model_step(owner, run_id, "prediction")
        return self.status(owner, run_id)

    def execute_models(self, owner, run_id):
        self._model_step(owner, run_id, "prediction")
        if self.status(owner, run_id)["state"] == "BLOCKED":
            return self.status(owner, run_id)
        self._model_step(owner, run_id, "policy")
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            self._live(db, run)
            if run["state"] == "POLICY_ACCEPTED":
                try:
                    _, _, manifest = self._evidence(db, run)
                except GateError as exc:
                    db.execute("UPDATE runs SET state='BLOCKED',reason=? WHERE id=?", (exc.code, run_id))
                    self._event(db, run_id, "engineering_safety_rejected", code=exc.code, action_executed=False)
                else:
                    db.execute("UPDATE runs SET state='SAFETY_ACCEPTED' WHERE id=?", (run_id,))
                    self._event(db, run_id, "engineering_safety_accepted", rule_version=manifest['rule_version'],
                                clinical_safety_validated=False, action_executed=False)
        return self.status(owner, run_id)

    def _evidence(self, db, run):
        self._live(db, run)
        snapshot = self._snapshot(db, run["snapshot_id"])
        prediction, p_hash = self._artifact(db, run, "prediction", completed_controller=True)
        policy, a_hash = self._artifact(db, run, "policy", completed_controller=True)
        profile = self._profile(db, run)
        (BasalPredictionOutput if profile else PredictionOutput).model_validate(prediction["payload"])
        action = (policy_contract(profile) if profile else PolicyOutput).model_validate(policy["payload"])
        if policy["parent_hash"] != p_hash or action.forecast_parent_hash != p_hash:
            raise GateError("FORECAST_PARENT_MISMATCH")
        if run["mode"] != "eval" or snapshot["source"] != "synthetic":
            raise GateError("RESEARCH_RULES_NOT_CONFIGURED")
        # Engineering contract limit, not a clinical dosing threshold.
        if profile:
            rate = rate_per_minute(action.action) if action.action is not None else None
        else:
            rate = action.action_value
        if action.status != "candidate" or rate is None or rate > (Fraction(profile['engineering_max_rate']) if profile else 0.05):
            raise GateError("ENGINEERING_SAFETY_REJECTED")
        manifest = {"run_id": run["id"], "snapshot_id": run["snapshot_id"], "snapshot_hash": digest(snapshot),
                    "prediction_hash": p_hash, "policy_hash": a_hash, "rule_version": RULE_VERSION,
                    "template_version": TEMPLATE_VERSION, "knowledge_hash": digest(TECHNICAL_NOTICE),
                    "kb_version": KB_VERSION}
        if profile:
            manifest.update(profile_hash=digest(profile), rule_version=profile['rule_version'],
                            template_version=profile['template_version'], expires_at=run['expires'],
                            decision_time=snapshot['decision_time'], time_basis=profile['time_basis'])
        return prediction, policy, manifest

    def create_draft_job(self, owner, run_id):
        token = secrets.token_urlsafe(32)
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            self._live(db, run)
            if run["state"] not in {"SAFETY_ACCEPTED", "DRAFT_PENDING", "DRAFT_READY", "REVIEWED", "REVIEW_BLOCKED"}:
                raise GateError("INVALID_STAGE")
            if run["revision"] >= 3:
                raise GateError("REVISION_BUDGET_EXHAUSTED")
            _, _, manifest = self._evidence(db, run)
            # Replacing the draft fences any pending proposal and reviewer attempts.
            db.execute("UPDATE jobs SET status='FENCED' WHERE run_id=? AND status='RUNNING'", (run_id,))
            job_id = self._new_job(db, run, "draft", digest(manifest), capability_hash=digest(token))
            db.execute("UPDATE jobs SET deadline=? WHERE id=?", (self.clock() + 300, job_id))
            db.execute("UPDATE runs SET state='DRAFT_PENDING',draft_id=NULL,reason=NULL WHERE id=?", (run_id,))
        return {"job_id": job_id, "capability": token, "evidence_hash": digest(manifest),
                "template_version": manifest['template_version'],
                "allowed_sections": ["forecast", "policy", "limitations"]}

    def submit_proposal(self, job_id, capability, raw):
        proposal = Proposal.model_validate(strict_json(raw))
        with self.store.tx() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=? AND step='draft'", (job_id,)).fetchone()
            if not job or not secrets.compare_digest(job["capability_hash"] or "", digest(capability)):
                raise GateError("INVALID_CAPABILITY", 403)
            run = db.execute("SELECT * FROM runs WHERE id=?", (job["run_id"],)).fetchone()
            self._current_job(db, run, job_id)
            if run["state"] != "DRAFT_PENDING":
                raise GateError("INVALID_STAGE")
            prediction, policy, manifest = self._evidence(db, run)
            evidence_hash = digest(manifest)
            if job["input_digest"] != evidence_hash:
                raise GateError("EVIDENCE_CHANGED")
            body = render(proposal, prediction, policy, evidence_hash, profile=self._profile(db, run))
            draft_id, revision = uid(), run["revision"] + 1
            db.execute("INSERT INTO drafts VALUES(?,?,?,?,?,?)",
                       (draft_id, run["id"], canonical(body), digest(body), evidence_hash, revision))
            db.execute("UPDATE jobs SET status='ACCEPTED' WHERE id=?", (job_id,))
            db.execute("UPDATE runs SET state='DRAFT_READY',draft_id=?,revision=? WHERE id=?",
                       (draft_id, revision, run["id"]))
            self._event(db, run["id"], "draft_created", draft_id=draft_id, report_hash=digest(body), revision=revision)
        # Do not return unreviewed medical-looking contents to an agent or UI.
        return {"draft_id": draft_id, "report_hash": digest(body), "state": "DRAFT_READY"}

    def _draft(self, db, run):
        row = db.execute("SELECT * FROM drafts WHERE id=? AND run_id=?", (run["draft_id"], run["id"])).fetchone()
        if not row:
            raise GateError("DRAFT_MISSING")
        body = json.loads(row["body"])
        if digest(body) != row["digest"]:
            raise GateError("DRAFT_INTEGRITY")
        if row['revision'] != run['revision']:
            raise GateError('DRAFT_REVISION_MISMATCH')
        prediction, policy, manifest = self._evidence(db, run)
        if row["evidence_hash"] != digest(manifest):
            raise GateError("EVIDENCE_CHANGED")
        proposal = Proposal(sections=[section["kind"] for section in body["sections"]])
        if render(proposal, prediction, policy, digest(manifest), profile=self._profile(db, run)) != body:
            raise GateError("TEMPLATE_MISMATCH")
        return row, body

    def review(self, owner, run_id, role):
        if role not in {"medical", "ethics"}:
            raise GateError("INVALID_REVIEWER", 422)
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            self._live(db, run)
            if run["state"] not in {"DRAFT_READY", "REVIEWED", "REVIEW_BLOCKED"}:
                raise GateError("INVALID_STAGE")
            draft, body = self._draft(db, run)
            executor = self._executor(run, role)
            old = db.execute("SELECT * FROM reviews WHERE draft_id=? AND role=?", (draft["id"], role)).fetchone()
            if old:
                checked = self._validate_review_record(db, run, draft, body, old)
                return {"role": role, "verdict": checked['verdict']}
            request = self._review_request(db, run, draft, body, role)
            job_id = self._new_job(db, run, "review_" + role, digest(request), draft_id=draft["id"])
            real_review = executor == REVIEW_EXECUTORS[role]
            if real_review:
                db.execute('UPDATE jobs SET deadline=? WHERE id=?', (self.clock() + self._review_agent.timeout, job_id))
        try:
            raw_output = (self._review_agent.execute(self, executor, request, owner, run_id, job_id, role)
                          if real_review else self._invoke(executor, request, owner, run_id, job_id))
            output = ReviewOutput.model_validate(raw_output).model_dump()
            if output["report_hash"] != draft["digest"] or output["evidence_hash"] != draft["evidence_hash"]:
                raise GateError("REVIEW_BINDING")
            with self.store.tx() as db:
                current = self._run(db, run_id, owner)
                self._current_job(db, current, job_id)
                if current["draft_id"] != draft["id"]:
                    raise GateError("STALE_REVIEW")
                current_draft, current_body = self._draft(db, current)
                job = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
                if (self._review_request(db, current, current_draft, current_body, role) != request
                        or job['input_digest'] != digest(request) or self._executor(current, role) != executor):
                    raise GateError('REVIEW_BINDING')
                envelope = {**output, "producer": executor.identity, "version": executor.version,
                            "origin": executor.origin, "role": role, "job_id": job_id}
                if real_review:
                    envelope['binding'] = request['binding']
                db.execute("INSERT INTO reviews(id,run_id,draft_id,role,verdict,body,job_id,digest) VALUES(?,?,?,?,?,?,?,?)",
                           (uid(), run_id, draft["id"], role, output["verdict"], canonical(envelope), job_id, digest(envelope)))
                db.execute("UPDATE jobs SET status='ACCEPTED' WHERE id=?", (job_id,))
                verdicts = [r[0] for r in db.execute("SELECT verdict FROM reviews WHERE draft_id=?", (draft["id"],))]
                state = "REVIEW_BLOCKED" if any(v != "pass" for v in verdicts) else "REVIEWED" if len(verdicts) == 2 else "DRAFT_READY"
                db.execute("UPDATE runs SET state=?,reason=? WHERE id=?",
                           (state, "REVIEW_REJECTED" if state == "REVIEW_BLOCKED" else None, run_id))
                self._event(db, run_id, "review_completed", role=role, verdict=output["verdict"],
                            draft_id=draft["id"], report_hash=draft["digest"], revision=current_draft['revision'],
                            evidence_hash=draft['evidence_hash'], reviewer_version=executor.version,
                            config_hash=request.get('binding', {}).get('config_hash'))
                return {"role": role, "verdict": output["verdict"], "state": state}
        except Exception as exc:
            code = exc.code if isinstance(exc, GateError) else "INVALID_REVIEW_OUTPUT"
            self._job_failure(owner, run_id, job_id, code)
            if isinstance(exc, GateError):
                raise
            raise GateError(code) from exc

    def _release_checks(self, db, run):
        self._live(db, run)
        if run["state"] not in {"REVIEWED", "RELEASED"}:
            raise GateError("REVIEW_REQUIRED")
        draft, body = self._draft(db, run)
        rows = db.execute("SELECT * FROM reviews WHERE draft_id=? AND run_id=?", (draft["id"], run["id"])).fetchall()
        if {r["role"] for r in rows} != {"medical", "ethics"}:
            raise GateError("REVIEW_REQUIRED")
        for row in rows:
            review = self._validate_review_record(db, run, draft, body, row)
            if review['verdict'] != 'pass' or review['issues']:
                raise GateError("REVIEW_INVALID")
        return draft, body, [r["id"] for r in rows]

    def release(self, owner, run_id):
        with self.store.tx() as db:
            run = self._run(db, run_id, owner)
            draft, body, reviews = self._release_checks(db, run)
            old = db.execute("SELECT id FROM releases WHERE run_id=?", (run_id,)).fetchone()
            if old:
                return {"release_id": old["id"], "state": "RELEASED"}
            release_id = uid()
            bundle = {"release_id": release_id, "run_id": run_id, "case_id": run["case_id"],
                      "snapshot_id": run["snapshot_id"], "mode": run["mode"], "report": body,
                      "report_hash": draft["digest"], "evidence_hash": draft["evidence_hash"],
                      "review_ids": reviews, "released_at": self.clock(), "schema_version": 1}
            db.execute("INSERT INTO releases VALUES(?,?,?,?)",
                       (release_id, run_id, canonical(bundle), digest(bundle)))
            db.execute("UPDATE runs SET state='RELEASED' WHERE id=?", (run_id,))
            self._event(db, run_id, "release_committed", release_id=release_id, release_hash=digest(bundle))
        return {"release_id": release_id, "state": "RELEASED"}

    def get_release(self, owner, release_id):
        with self.store.tx() as db:
            row = db.execute("SELECT * FROM releases WHERE id=?", (release_id,)).fetchone()
            if not row:
                raise GateError("NOT_FOUND", 404)
            run = self._run(db, row["run_id"], owner)
            draft, body, reviews = self._release_checks(db, run)
            bundle = json.loads(row["body"])
            if (digest(bundle) != row["digest"] or bundle["report"] != body
                    or bundle["report_hash"] != draft["digest"] or set(bundle["review_ids"]) != set(reviews)):
                raise GateError("RELEASE_INTEGRITY")
            return bundle

    def events(self, owner, run_id):
        with self.store.tx() as db:
            self._run(db, run_id, owner)
            return [{**dict(row), "body": json.loads(row["body"])} for row in
                    db.execute("SELECT * FROM events WHERE run_id=? ORDER BY sequence", (run_id,))]

    def recover_interrupted(self):
        """Call once at exclusive service startup, never while another Core is running."""
        with self.store.tx() as db:
            jobs = db.execute("SELECT * FROM jobs WHERE status='RUNNING'").fetchall()
            for job in jobs:
                db.execute("UPDATE jobs SET status='FENCED' WHERE id=?", (job["id"],))
                state = "SAFETY_ACCEPTED" if job["step"] == "draft" else "REVIEW_BLOCKED" if job["step"].startswith("review_") else "UNAVAILABLE"
                db.execute("UPDATE runs SET state=?,reason='SERVICE_RESTARTED' WHERE id=? AND state NOT IN ('CANCELLED','INVALIDATED')",
                           (state, job["run_id"]))
                self._event(db, job["run_id"], "interrupted_job_fenced", job_id=job["id"])
            return len(jobs)

    def deliver_outbox(self, sink, limit=100):
        """Trusted bridge callback. Ack only after sink success; delivery is at-least-once.

        A sink must deduplicate on event id. No agent can choose the sink URL.
        """
        with self.store.tx() as db:
            rows = [dict(row) for row in db.execute("""SELECT e.* FROM events e JOIN outbox o
                ON o.event_id=e.id WHERE o.delivered=0 ORDER BY e.sequence LIMIT ?""", (limit,))]
        delivered = 0
        for row in rows:
            sink({**row, "body": json.loads(row["body"])})
            with self.store.tx() as db:
                db.execute("UPDATE outbox SET delivered=1 WHERE event_id=?", (row["id"],))
            delivered += 1
        return delivered
