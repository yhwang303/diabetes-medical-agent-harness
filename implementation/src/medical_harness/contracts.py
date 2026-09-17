"""Strict boundary types. No client field can confer evidence authority."""

import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GateError(Exception):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code)
        self.code = code
        self.status = status


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(raw: str | bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("non-finite JSON number")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    if len(raw) > 131072:
        raise GateError("PAYLOAD_TOO_LARGE", 413)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise GateError("INVALID_JSON", 422) from exc


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


Number = Annotated[float, Field(allow_inf_nan=False)]
Origin = Literal["model", "simulator", "fixture"]


class Point(StrictModel):
    time: str
    value: Annotated[Number | None, Field(ge=0, le=1000)]


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone offset required")
    return parsed


class SnapshotInput(StrictModel):
    cohort: Literal["adult_t1d"]
    source: Literal["synthetic", "historical", "simulation"]
    source_ref: Annotated[str, Field(min_length=1, max_length=200)]
    decision_time: str
    unit: Literal["mg/dL"]
    sampling_minutes: Literal[5]
    history: Annotated[list[Point], Field(min_length=6, max_length=288)]
    missing_mask: list[bool]

    @model_validator(mode="after")
    def chronology(self):
        end = timestamp(self.decision_time)
        times = [timestamp(p.time) for p in self.history]
        if times[-1] != end:
            raise ValueError("last observation must match decision_time")
        if any((b - a).total_seconds() != 300 for a, b in zip(times, times[1:])):
            raise ValueError("history must be an ordered 5-minute grid")
        if self.missing_mask != [p.value is None for p in self.history]:
            raise ValueError("missing mask must match explicit nulls")
        return self


class PredictionOutput(StrictModel):
    input_digest: str
    status: Literal["ok"]
    unit: Literal["mg/dL"]
    interval_minutes: Literal[5]
    values: Annotated[list[Annotated[Number, Field(ge=0, le=1000)]], Field(min_length=6, max_length=6)]
    uncertainty: Literal["unavailable"]


class PolicyOutput(StrictModel):
    input_digest: str
    forecast_parent_hash: str
    status: Literal["candidate", "abstain", "unsupported", "error"]
    action_kind: Literal["basal_rate"]
    action_unit: Literal["U/min"]
    action_value: Annotated[Number | None, Field(ge=0, le=1)]
    duration_minutes: Literal[5]
    reason: Literal["fixture_only", "insufficient_input", "unsupported", "model_error"]

    @model_validator(mode="after")
    def candidate_only(self):
        if (self.status == "candidate") != (self.action_value is not None):
            raise ValueError("only a candidate may carry an action")
        expected = {"candidate": "fixture_only", "abstain": "insufficient_input",
                    "unsupported": "unsupported", "error": "model_error"}
        if self.reason != expected[self.status]:
            raise ValueError("reason must agree with policy status")
        return self


class Proposal(StrictModel):
    """Agent selects supported sections, never supplies prose, numbers or HTML."""

    sections: Annotated[list[Literal["forecast", "policy", "limitations"]], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def required_sections(self):
        if set(self.sections) != {"forecast", "policy", "limitations"}:
            raise ValueError("all required sections must appear exactly once")
        return self


class ReviewOutput(StrictModel):
    verdict: Literal["pass", "fail", "abstain"]
    report_hash: str
    evidence_hash: str
    issues: list[Literal["invalid_content", "fixture_review_failure", "unsupported_claim", "evidence_mismatch",
                         "missing_limitations", "privacy_risk", "unsafe_action", "review_uncertain", "instruction_injection"]]

    @model_validator(mode="after")
    def no_unresolved_pass(self):
        if self.verdict == "pass" and self.issues:
            raise ValueError("pass cannot contain unresolved issues")
        return self


class ErrorResponse(StrictModel):
    # Fixed metadata only: never echo raw input, model output or credentials.
    error: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=100)]


FixtureScenario = Literal["candidate", "abstain", "unsupported", "error", "invalid_output", "parent_mismatch", "safety_rejected", "timeout"]


class RunRequest(StrictModel):
    case_id: Annotated[str, Field(min_length=1, max_length=64)]
    mode: Literal["eval", "research"] = "research"
    idempotency_key: Annotated[str, Field(min_length=1, max_length=128)]
    expected_snapshot_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")] | None = None
    fixture_scenario: FixtureScenario | None = None


class ImportRequest(StrictModel):
    file_name: Annotated[str, Field(min_length=1, max_length=200)]
    content: Annotated[str, Field(min_length=1, max_length=65536)]

    @model_validator(mode="after")
    def file_boundary(self):
        try:
            self.file_name.encode("utf-8")
            encoded = self.content.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("valid UTF-8 required") from exc
        if not self.file_name.lower().endswith(".json") or any(c in self.file_name for c in "/\\"):
            raise ValueError("a JSON filename without a path is required")
        if any(ord(c) < 32 or ord(c) == 127 for c in self.file_name):
            raise ValueError("control characters forbidden")
        if len(encoded) > 65536:
            raise ValueError("file exceeds 64 KiB")
        return self


class PredictionRequest(StrictModel):
    snapshot: SnapshotInput
    snapshot_id: str
    run_id: str
    input_digest: str


class PolicyRequest(PredictionRequest):
    prediction: PredictionOutput
    prediction_hash: str


class ArtifactEnvelope(StrictModel):
    id: str
    schema_version: Literal[1]
    kind: Literal["prediction", "policy"]
    run_id: str
    case_id: str
    snapshot_id: str
    job_id: str
    attempt: Annotated[int, Field(ge=1, le=3)]
    producer: str
    version: str
    origin: Origin
    input_digest: str
    parent_hash: str | None
    payload: PredictionOutput | PolicyOutput
    accepted_at: Number

    @model_validator(mode="after")
    def payload_kind(self):
        if (self.kind == "prediction") != isinstance(self.payload, PredictionOutput):
            raise ValueError("payload must match artifact kind")
        if (self.kind == "prediction") != (self.parent_hash is None):
            raise ValueError("only policy has a forecast parent")
        return self


class EvidenceCard(StrictModel):
    kind: Literal["input", "prediction", "policy", "safety", "reviews", "release"]
    status: Literal["verified", "waiting", "blocked"]
    reason: str | None = None
    reference_id: str | None = None
    digest: str | None = None
    producer: str | None = None
    version: str | None = None
    origin: Origin | None = None


class CaseEvidence(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    snapshot_id: str
    source: Literal["synthetic", "historical", "simulation"]
    missing_count: int
    run_id: str | None
    run_state: str
    mode: Literal["eval", "research"] | None
    fixture_scenario: FixtureScenario | None = None
    checked_at: Number
    release_allowed: bool
    clinical_validation: Literal[False] = False
    cards: list[EvidenceCard]


class AgentTaskRequest(StrictModel):
    case_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    snapshot_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    idempotency_key: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]


def contract_catalog() -> dict:
    """Versioned public schemas, not a result registration or trust interface."""
    from .basal_inputs import BasalResearchInput, input_profile_catalog
    from .data_policy import DataPermissionRequest, data_policy_catalog
    from .data_lifecycle import DeleteCaseRequest
    from .prediction_agent import PredictionJobQuery
    from .rl_agent import RLJobQuery
    from .basal_actions import (BasalAction, BasalPolicyOutput, BasalPredictionOutput,
                               BasalPredictionRequest, BasalPolicyRequest, BasalArtifactEnvelope, fixture_profile, legacy_fixture_profile,
                               BasalRateAction, BasalRatePolicyOutput, BasalRateArtifactEnvelope)
    models = (SnapshotInput, PredictionRequest, PredictionOutput, PolicyRequest, PolicyOutput,
              ArtifactEnvelope, Proposal, ReviewOutput, RunRequest, ImportRequest, ErrorResponse,
              EvidenceCard, CaseEvidence, AgentTaskRequest, BasalResearchInput, BasalAction,
              BasalPolicyOutput, BasalPredictionOutput, BasalPredictionRequest, BasalPolicyRequest, BasalArtifactEnvelope,
              BasalRateAction, BasalRatePolicyOutput, BasalRateArtifactEnvelope, DataPermissionRequest, DeleteCaseRequest, PredictionJobQuery, RLJobQuery)
    return {"contract_version": "harness-engineering-v1", "clinical_use": False,
            "input_profiles": input_profile_catalog(), "data_policy": data_policy_catalog(),
            "engineering_execution_profiles": {legacy_fixture_profile()['version']: legacy_fixture_profile(),
                                               fixture_profile()['version']: fixture_profile()},
            "schemas": {model.__name__: model.model_json_schema() for model in models}}
