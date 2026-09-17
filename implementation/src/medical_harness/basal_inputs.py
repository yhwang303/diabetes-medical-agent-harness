"""Versioned research facts; declarations are not authenticated model evidence."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import GateError, Number, SnapshotInput, StrictModel, timestamp


INPUT_VERSION = 'basal-research-input-v1'
USE_PROFILE = 'adult-t1d-basal-simulation-v1'
Reference = Annotated[str, Field(min_length=1, max_length=200)]


class ResearchSource(StrictModel):
    subject_ref: Reference
    source_ref: Reference
    source_version: Reference
    locator: Reference
    kind: Literal['synthetic', 'simulator']
    available_at: str


class ResearchTreatment(StrictModel):
    insulin: Literal['generic_simulation_insulin']
    route: Literal['subcutaneous_pump']
    regimen: Literal['basal_only']
    provenance: ResearchSource


class BasalDelivery(StrictModel):
    record_id: Reference
    record_kind: Literal['actual_delivery']
    start_time: str
    end_time: str
    delivered_units: Annotated[Number, Field(ge=0)]
    unit: Literal['U']

    @model_validator(mode='after')
    def interval(self):
        if timestamp(self.start_time) >= timestamp(self.end_time):
            raise ValueError('delivery requires a positive elapsed interval')
        return self


class ResearchInsulinHistory(StrictModel):
    coverage_start: str
    coverage_end: str
    deliveries: Annotated[list[BasalDelivery], Field(min_length=1, max_length=288)]
    provenance: ResearchSource

    @model_validator(mode='after')
    def coverage(self):
        start, end = timestamp(self.coverage_start), timestamp(self.coverage_end)
        if start >= end or end > timestamp(self.provenance.available_at):
            raise ValueError('invalid delivery coverage or availability')
        if len({r.record_id for r in self.deliveries}) != len(self.deliveries):
            raise ValueError('duplicate delivery record')
        cursor = start
        for record in self.deliveries:
            if timestamp(record.start_time) != cursor:
                raise ValueError('delivery intervals must cover the declared window without gaps or overlap')
            cursor = timestamp(record.end_time)
        if cursor != end:
            raise ValueError('delivery coverage must end at the declared cutoff')
        return self


class BasalResearchInput(SnapshotInput):
    # The inherited CGM grid is an input storage boundary, not a verified model window.
    contract_version: Literal['basal-research-input-v1']
    use_profile: Literal['adult-t1d-basal-simulation-v1']
    source: Literal['synthetic', 'simulation']
    provenance: ResearchSource
    treatment: ResearchTreatment | None
    insulin_history: ResearchInsulinHistory | None

    @model_validator(mode='after')
    def research_facts(self):
        decision = timestamp(self.decision_time)
        expected = 'synthetic' if self.source == 'synthetic' else 'simulator'
        sources = [self.provenance]
        if self.treatment is not None:
            sources.append(self.treatment.provenance)
        if self.insulin_history is not None:
            sources.append(self.insulin_history.provenance)
            if timestamp(self.insulin_history.coverage_end) != decision:
                raise ValueError('insulin coverage must end at decision time')
        if self.provenance.source_ref != self.source_ref:
            raise ValueError('CGM source references disagree')
        for source in sources:
            if source.kind != expected or source.subject_ref != self.provenance.subject_ref:
                raise ValueError('research source or subject mismatch')
            if timestamp(source.available_at) > decision:
                raise ValueError('future information is not available at decision time')
        if timestamp(self.provenance.available_at) < timestamp(self.history[-1].time):
            raise ValueError('CGM availability precedes its last observation')
        return self


def parse_snapshot(raw):
    # Explicit dispatch: unknown versions fail, never retry them as legacy fixtures.
    model = BasalResearchInput if isinstance(raw, dict) and 'contract_version' in raw else SnapshotInput
    return model.model_validate(raw)


def input_block_reason(snapshot):
    if any(snapshot['missing_mask']):
        return 'MISSING_INPUT'
    if 'contract_version' in snapshot:
        if snapshot['treatment'] is None:
            return 'TREATMENT_REQUIRED'
        if snapshot['insulin_history'] is None:
            return 'INSULIN_HISTORY_REQUIRED'
        # P01b/M01/M02 must bind actual validated consumers before this can execute.
        return 'INPUT_PROFILE_NOT_CONFIGURED'
    return None


def require_execution_profile(snapshot):
    if 'contract_version' in snapshot:
        raise GateError(input_block_reason(snapshot))


def input_profile_catalog():
    return {USE_PROFILE: {
        'input_contract': INPUT_VERSION, 'clinical_use': False,
        'execution_status': 'not_configured', 'allowed_sources': ['synthetic', 'simulation'],
        'cohort': 'adult_t1d', 'action_scope': 'pump_basal_infusion',
        'prediction_window': None, 'required_insulin_history_minutes': None,
        'consumers': {
            'history': ['prediction', 'policy', 'input_checks'],
            'treatment': ['applicability_checks', 'policy'],
            'insulin_history': ['policy', 'input_checks'],
            'provenance': ['source_checks'],
        },
        'consumer_status': 'planned_only_no_model_or_medical_rule_binding',
    }}
