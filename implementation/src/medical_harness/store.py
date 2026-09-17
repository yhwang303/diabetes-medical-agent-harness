"""Private SQLite ledger; domain state and durable outbox commit together."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .paths import confined


SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, owner TEXT NOT NULL, snapshot_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, case_id TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS imports(
 case_id TEXT PRIMARY KEY REFERENCES cases(id), snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
 file_name TEXT NOT NULL, file_hash TEXT NOT NULL, byte_count INTEGER NOT NULL,
 imported_at REAL NOT NULL, UNIQUE(case_id,snapshot_id));
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, case_id TEXT NOT NULL, snapshot_id TEXT NOT NULL, owner TEXT NOT NULL,
 mode TEXT NOT NULL, state TEXT NOT NULL, reason TEXT, expires REAL NOT NULL, versions TEXT NOT NULL,
 draft_id TEXT, revision INTEGER NOT NULL DEFAULT 0,
 idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, UNIQUE(owner,idempotency_key));
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step TEXT NOT NULL, attempt INTEGER NOT NULL,
 status TEXT NOT NULL, deadline REAL NOT NULL, input_digest TEXT NOT NULL,
 capability_hash TEXT, draft_id TEXT, UNIQUE(run_id,step,attempt));
CREATE TABLE IF NOT EXISTS artifacts(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL,
 digest TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE, UNIQUE(run_id,kind));
CREATE TABLE IF NOT EXISTS drafts(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL,
 evidence_hash TEXT NOT NULL, revision INTEGER NOT NULL, UNIQUE(run_id,revision));
CREATE TABLE IF NOT EXISTS reviews(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL, draft_id TEXT NOT NULL, role TEXT NOT NULL,
 verdict TEXT NOT NULL, body TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE,
 UNIQUE(draft_id,role));
CREATE TABLE IF NOT EXISTS releases(
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, body TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions(name TEXT PRIMARY KEY, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS review_configs(
 run_id TEXT NOT NULL REFERENCES runs(id), role TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL,
 PRIMARY KEY(run_id,role));
CREATE TABLE IF NOT EXISTS run_profiles(
 run_id TEXT PRIMARY KEY REFERENCES runs(id), body TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agent_tasks(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, case_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
 run_id TEXT NOT NULL UNIQUE REFERENCES runs(id), idempotency_key TEXT NOT NULL,
 state TEXT NOT NULL, phase TEXT NOT NULL, reason TEXT, created_at REAL NOT NULL,
 UNIQUE(owner,idempotency_key));
CREATE TABLE IF NOT EXISTS data_permissions(
 case_id TEXT NOT NULL REFERENCES cases(id), purpose TEXT NOT NULL, snapshot_id TEXT NOT NULL,
 allowed INTEGER NOT NULL, revision INTEGER NOT NULL, policy_hash TEXT NOT NULL, instance TEXT NOT NULL,
 PRIMARY KEY(case_id,purpose));
CREATE TABLE IF NOT EXISTS job_data_permissions(
 job_id TEXT PRIMARY KEY REFERENCES jobs(id), body TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS job_dependencies(
 child_job_id TEXT PRIMARY KEY REFERENCES jobs(id), parent_job_id TEXT NOT NULL REFERENCES jobs(id));
CREATE TABLE IF NOT EXISTS case_deletions(
 case_id TEXT PRIMARY KEY REFERENCES cases(id), requested_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sdk_sessions(
 id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id), run_id TEXT NOT NULL REFERENCES runs(id),
 job_id TEXT NOT NULL REFERENCES jobs(id), ledger_scope TEXT NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, run_id TEXT,
 kind TEXT NOT NULL, body TEXT NOT NULL, time REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(
 event_id TEXT PRIMARY KEY REFERENCES events(id), delivered INTEGER NOT NULL DEFAULT 0);
"""


class Store:
    def __init__(self, path: Path):
        path = confined(path)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.tx() as db:
            db.executescript(SCHEMA)
        # Additive migration preserves existing runs, artifacts and idempotency hashes.
        with self.tx() as db:
            if "fixture_scenario" not in {row["name"] for row in db.execute("PRAGMA table_info(runs)")}:
                db.execute("ALTER TABLE runs ADD COLUMN fixture_scenario TEXT")
            if "digest" not in {row["name"] for row in db.execute("PRAGMA table_info(reviews)")}:
                db.execute("ALTER TABLE reviews ADD COLUMN digest TEXT")
        path.chmod(0o600)

    @contextmanager
    def tx(self):
        # Check SQLite's own write targets as well as the main database.
        for suffix in ("", "-wal", "-shm", "-journal"):
            confined(Path(str(self.path) + suffix))
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
