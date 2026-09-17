"""Basal research action semantics and a frozen, explicitly nonclinical fixture profile."""

from datetime import timedelta
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .basal_inputs import BasalResearchInput, INPUT_VERSION, USE_PROFILE, input_block_reason
from .contracts import (ArtifactEnvelope, GateError, Number, PredictionOutput,
                        StrictModel, timestamp)

LEGACY_PROFILE_VERSION = 'basal-fixture-profile-v1'
PROFILE_VERSION = 'basal-fixture-profile-v2'
TEMPLATE_VERSION = 'basal-fixture-template-v2'
RULE_VERSION = 'basal-fixture-rules-v2'
CONVERSION_VERSION = 'basal-rate-conversion-v1'


class BasalAction(StrictModel):
    action_kind: Literal['basal_rate']
    insulin: Literal['generic_simulation_insulin']
    route: Literal['subcutaneous_pump']
    action_unit: Literal['U/min']
    action_value: Annotated[Number, Field(ge=0, le=1)]
    duration_minutes: Literal[5]
    decision_time: str
    end_time: str

    @model_validator(mode='after')
    def semantics(self):
        if timestamp(self.end_time) - timestamp(self.decision_time) != timedelta(minutes=self.duration_minutes):
            raise ValueError('action end must match duration')
        # This is an engineering representation quantum, not a validated pump resolution.
        if Decimal(str(self.action_value)) % Decimal('0.000001') != 0:
            raise ValueError('unsupported action precision; never silently round')
        return self


class BasalPredictionOutput(PredictionOutput):
    profile_hash: str


class BasalPolicyOutput(StrictModel):
    schema_version: Literal['basal-policy-v1']
    input_digest: str
    forecast_parent_hash: str
    profile_hash: str
    status: Literal['candidate', 'abstain', 'unsupported', 'error']
    reason: Literal['fixture_only', 'insufficient_input', 'unsupported', 'model_error']
    action: BasalAction | None

    @model_validator(mode='after')
    def candidate_only(self):
        expected = {'candidate': 'fixture_only', 'abstain': 'insufficient_input',
                    'unsupported': 'unsupported', 'error': 'model_error'}
        if (self.status == 'candidate') != (self.action is not None) or self.reason != expected[self.status]:
            raise ValueError('only candidate carries an action; failure must not become zero')
        return self


class BasalPredictionRequest(StrictModel):
    snapshot: BasalResearchInput
    snapshot_id: str
    run_id: str
    input_digest: str
    profile_hash: str


class BasalPolicyRequest(BasalPredictionRequest):
    prediction: BasalPredictionOutput
    prediction_hash: str


class BasalArtifactEnvelope(ArtifactEnvelope):
    payload: BasalPredictionOutput | BasalPolicyOutput


class BasalRateAction(BasalAction):
    action_unit: Literal['U/min', 'U/h']
    action_value: Annotated[Number, Field(ge=0, le=60)]

    @model_validator(mode='after')
    def semantics(self):
        if timestamp(self.end_time) - timestamp(self.decision_time) != timedelta(minutes=self.duration_minutes):
            raise ValueError('action end must match duration')
        if rate_per_minute(self) > 1:
            raise ValueError('action exceeds engineering representation bound')
        return self


class BasalRatePolicyOutput(BasalPolicyOutput):
    schema_version: Literal['basal-policy-v2']
    action: BasalRateAction | None


class BasalRateArtifactEnvelope(ArtifactEnvelope):
    payload: BasalPredictionOutput | BasalRatePolicyOutput


def rate_per_minute(action):
    # Exact with respect to the model's JSON number; no float division or quantization.
    return Fraction(str(action.action_value)) / (60 if action.action_unit == 'U/h' else 1)


def policy_contract(profile):
    return BasalRatePolicyOutput if profile['version'] == PROFILE_VERSION else BasalPolicyOutput


def artifact_contract(profile):
    return BasalRateArtifactEnvelope if profile['version'] == PROFILE_VERSION else BasalArtifactEnvelope


def legacy_fixture_profile():
    # Returned fresh for each comparison. No caller-provided configuration enables a model.
    return {'version': LEGACY_PROFILE_VERSION, 'input_contract': INPUT_VERSION, 'use_profile': USE_PROFILE,
            'mode': 'eval', 'source': 'synthetic', 'origin': 'fixture', 'clinical_use': False,
            'time_basis': 'synthetic_replay', 'max_wall_seconds': 300,
            'action_minutes': 5, 'rate_quantum': '0.000001', 'engineering_max_rate': '0.05',
            'input_minimum': {'cgm_points': 6, 'insulin_history_minutes': 25},
            'prediction': {'identity': 'fixture.basal-predictor', 'version': 'basal-prediction-v1',
                           'interval_minutes': 5, 'points': 6, 'uncertainty': 'unavailable',
                           'preprocessing': 'no-transform-v1', 'reads': ['history[-1].value']},
            'policy': {'identity': 'fixture.basal-policy', 'version': 'basal-policy-v1',
                       'preprocessing': 'no-transform-v1', 'training_dependencies': [],
                       'behavior': 'constant_fixture_not_trained',
                       'reads': ['decision_time', 'input_digest', 'prediction_hash', 'profile_hash']},
            'input_checks': ['cohort', 'source', 'use_profile', 'history', 'missing_mask',
                             'treatment', 'insulin_history', 'provenance'],
            'rule_version': 'basal-fixture-rules-v1', 'template_version': 'basal-fixture-template-v1',
            'real_model_configuration': None, 'medical_validation': False}


def fixture_profile():
    profile = legacy_fixture_profile()
    profile.update(version=PROFILE_VERSION, rule_version=RULE_VERSION, template_version=TEMPLATE_VERSION,
                   rate_quantum=None, unit_conversion={'version': CONVERSION_VERSION,
                       'accepted_units': ['U/min', 'U/h'], 'canonical_unit': 'U/min',
                       'arithmetic': 'exact_rational', 'source_preserved': True,
                       'display_significant_digits': 28, 'display_rounding': 'half_even', 'approximation_marked': True})
    profile['policy'].update(identity='fixture.basal-rate-policy', version='basal-policy-v2')
    return profile


def check_fixture_input(snapshot, mode):
    value = BasalResearchInput.model_validate(snapshot)
    if mode != 'eval':
        raise GateError('INPUT_PROFILE_NOT_CONFIGURED')
    if value.source != 'synthetic':
        raise GateError('FIXTURE_REQUIRES_SYNTHETIC_CASE')
    reason = input_block_reason(snapshot)
    if reason != 'INPUT_PROFILE_NOT_CONFIGURED':
        raise GateError(reason)
    minimum = fixture_profile()['input_minimum']
    coverage = timestamp(value.insulin_history.coverage_end) - timestamp(value.insulin_history.coverage_start)
    if len(value.history) < minimum['cgm_points'] or coverage.total_seconds() < minimum['insulin_history_minutes'] * 60:
        raise GateError('INSUFFICIENT_INPUT_WINDOW')


def check_output_binding(snapshot, output, profile_hash, step):
    if output['profile_hash'] != profile_hash:
        raise GateError('MODEL_PROFILE_BINDING')
    if step == 'policy' and output['status'] == 'candidate':
        if timestamp(output['action']['decision_time']) != timestamp(snapshot['decision_time']):
            raise GateError('ACTION_TIME_BINDING')


def display_action(raw):
    action = BasalAction.model_validate(raw)
    rate = Decimal(str(action.action_value))
    return {'action_kind': action.action_kind, 'insulin': action.insulin, 'route': action.route,
            'rate_u_per_min': format(rate, 'f'), 'rate_u_per_hour': format(rate * 60, 'f'),
            'duration_minutes': action.duration_minutes,
            'interval_total_units': format(rate * action.duration_minutes, 'f'),
            'decision_time': action.decision_time, 'end_time': action.end_time,
            'time_basis': 'synthetic_replay', 'action_executed': False,
            'notice': '合成研究区间的基础输注速率和区间总量；不是单次注射剂量，不外推为全天方案。'}


def display_rate_action(raw):
    action = BasalRateAction.model_validate(raw)
    rate = rate_per_minute(action)
    values = {'rate_u_per_min': rate, 'rate_u_per_hour': rate * 60,
              'interval_total_units': rate * action.duration_minutes}
    shown, exact, approximate = {}, {}, []
    for key, value in values.items():
        denominator = value.denominator
        for factor in (2, 5):
            while denominator % factor == 0:
                denominator //= factor
        with localcontext() as context:
            context.rounding = ROUND_HALF_EVEN
            context.prec = 28 if denominator != 1 else max(28, len(str(value.numerator)) + len(str(value.denominator)) + 6)
            shown[key] = format(Decimal(value.numerator) / Decimal(value.denominator), 'f')
        exact[key] = {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
        if denominator != 1:
            approximate.append(key)
    return {**shown, 'action_kind': action.action_kind, 'insulin': action.insulin, 'route': action.route,
            'duration_minutes': action.duration_minutes, 'decision_time': action.decision_time,
            'end_time': action.end_time, 'time_basis': 'synthetic_replay', 'action_executed': False,
            'conversion': {'version': CONVERSION_VERSION, 'source_value': action.action_value,
                           'source_unit': action.action_unit, 'exact': exact, 'approximate_fields': approximate},
            'notice': '合成研究基础输注；换算保留原始输出。标记为 approximate_fields 的数值仅为近似展示，精确值见分数；不是单次注射剂量，不外推全天。'}
