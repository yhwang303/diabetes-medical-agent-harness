"""Trusted executor seams. Real model bindings intentionally remain absent."""

from dataclasses import dataclass
from functools import partial
import time
from typing import get_args
from typing import Callable

from .contracts import FixtureScenario, GateError, Origin


@dataclass(frozen=True)
class Executor:
    identity: str
    version: str
    origin: Origin
    call: Callable[[dict], dict]


def fixture_prediction(request: dict) -> dict:
    # Persistence is only a deterministic contract fixture, not a validated predictor.
    last = request["snapshot"]["history"][-1]["value"]
    if last is None:
        raise GateError("MISSING_OBSERVATION")
    return {"input_digest": request["input_digest"], "status": "ok", "unit": "mg/dL",
            "interval_minutes": 5, "values": [last] * 6, "uncertainty": "unavailable"}


def fixture_policy(request: dict) -> dict:
    # Arbitrary fixed engineering action; never an insulin recommendation.
    return {"input_digest": request["input_digest"], "forecast_parent_hash": request["prediction_hash"],
            "status": "candidate", "action_kind": "basal_rate", "action_unit": "U/min",
            "action_value": 0.01, "duration_minutes": 5, "reason": "fixture_only"}


def fixture_review(request: dict) -> dict:
    body = request["report"]
    valid = (body["report_type"] == "EngineeringFixtureReport"
             and body["clinical_use"] is False and body["origin"] == "fixture"
             and body["warning"] == "工程测试替身：非真实预测、非真实RL、非医学审核，不用于用药。")
    return {"verdict": "pass" if valid else "fail", "report_hash": request["report_hash"],
            "evidence_hash": request["evidence_hash"], "issues": [] if valid else ["invalid_content"]}


def fixture_registry() -> dict[str, Executor]:
    return {
        "prediction": Executor("fixture.predictor", "fixture-prediction-v1", "fixture", fixture_prediction),
        "policy": Executor("fixture.policy", "fixture-policy-v1", "fixture", fixture_policy),
        "medical": Executor("fixture.medical-review", "fixture-medical-v1", "fixture", fixture_review),
        "ethics": Executor("fixture.ethics-review", "fixture-ethics-v1", "fixture", fixture_review),
    }


def fixture_policy_scenario(request: dict, *, scenario: str, timeout: float) -> dict:
    """Fixed trusted faults. These outputs still pass through all ordinary Core gates."""
    output = fixture_policy(request)
    if scenario in {"abstain", "unsupported", "error"}:
        output.update(status=scenario, action_value=None,
                      reason={"abstain": "insufficient_input", "unsupported": "unsupported", "error": "model_error"}[scenario])
    elif scenario == "invalid_output":
        output["action_unit"] = "invalid-fixture-unit"
    elif scenario == "parent_mismatch":
        output["forecast_parent_hash"] = "0" * 64
    elif scenario == "safety_rejected":
        output["action_value"] = 0.1  # Exceeds the existing arbitrary engineering bound, not a medical limit.
    elif scenario == "timeout":
        time.sleep(timeout + 0.1)  # A real late return; Core must discard it, not fabricate a timeout status.
    return output


def policy_scenario_registry(timeout: float) -> dict[str, Executor]:
    return {name: Executor(f"fixture.policy.{name}", f"fixture-policy-{name}-v1", "fixture",
                          partial(fixture_policy_scenario, scenario=name, timeout=timeout))
            for name in get_args(FixtureScenario)}


def basal_fixture_prediction(request):
    return {**fixture_prediction(request), 'profile_hash': request['profile_hash']}


def basal_fixture_policy(request):
    from datetime import timedelta
    from .contracts import timestamp
    decision = request['snapshot']['decision_time']
    return {'schema_version': 'basal-policy-v1', 'input_digest': request['input_digest'],
            'forecast_parent_hash': request['prediction_hash'], 'profile_hash': request['profile_hash'],
            'status': 'candidate', 'reason': 'fixture_only',
            'action': {'action_kind': 'basal_rate', 'insulin': 'generic_simulation_insulin',
                       'route': 'subcutaneous_pump', 'action_unit': 'U/min', 'action_value': 0.01,
                       'duration_minutes': 5, 'decision_time': decision,
                       'end_time': (timestamp(decision) + timedelta(minutes=5)).isoformat()}}


def legacy_basal_fixture_registry():
    return {'prediction': Executor('fixture.basal-predictor', 'basal-prediction-v1', 'fixture', basal_fixture_prediction),
            'policy': Executor('fixture.basal-policy', 'basal-policy-v1', 'fixture', basal_fixture_policy)}


def basal_rate_fixture_policy(request):
    output = basal_fixture_policy(request)
    output['schema_version'] = 'basal-policy-v2'
    output['action'].update(action_unit='U/h', action_value=0.6)
    return output


def basal_fixture_registry():
    return {'prediction': Executor('fixture.basal-predictor', 'basal-prediction-v1', 'fixture', basal_fixture_prediction),
            'policy': Executor('fixture.basal-rate-policy', 'basal-policy-v2', 'fixture', basal_rate_fixture_policy)}


# No fallback to fixtures in research mode. Future real adapters must be explicitly
# registered by trusted startup configuration and pass separate applicability checks.
REAL_EXECUTORS: dict[str, Executor] = {}
