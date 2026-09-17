"""Unit equivalence preserves the registered source and all release gates."""

from dataclasses import replace
from fractions import Fraction
import json

import pytest
from pydantic import ValidationError

from medical_harness.adapters import basal_rate_fixture_policy
from medical_harness.basal_actions import (BasalAction, CONVERSION_VERSION, LEGACY_PROFILE_VERSION,
                                          legacy_fixture_profile)
from medical_harness.contracts import GateError, canonical, digest
from test_basal_actions import basal_case, setup, release, count


def set_rate(core, value, unit):
    def call(request):
        output = basal_rate_fixture_policy(request)
        output['action'].update(action_value=value, action_unit=unit)
        return output
    core._basal_fixtures['policy'] = replace(core._basal_fixtures['policy'], call=call)


@pytest.mark.parametrize('value,unit,expected', [(0.6, 'U/h', Fraction(1, 100)),
    (0.01, 'U/min', Fraction(1, 100)), (0.1, 'U/h', Fraction(1, 600)),
    (0, 'U/h', Fraction(0)), (3, 'U/h', Fraction(1, 20)),
    (0.0000001, 'U/min', Fraction(1, 10000000)), (0.000001, 'U/h', Fraction(1, 60000000))])
def test_equivalent_rate_reaches_release_without_quantization(setup, value, unit, expected):
    core, run = setup
    set_rate(core, value, unit)
    released = release(core, run)
    report = core.get_release('alice', released['release_id'])['report']
    slot = report['sections'][1]
    assert slot['data']['action']['action_value'] == value
    assert slot['data']['action']['action_unit'] == unit
    conversion = slot['action_display']['conversion']
    assert (conversion['source_value'], conversion['source_unit']) == (value, unit)
    for key, factor in [('rate_u_per_min', 1), ('rate_u_per_hour', 60), ('interval_total_units', 5)]:
        exact = conversion['exact'][key]
        assert Fraction(int(exact['numerator']), int(exact['denominator'])) == expected * factor
        if key not in conversion['approximate_fields']:
            assert Fraction(slot['action_display'][key]) == expected * factor
    if value == 0.1:
        assert set(conversion['approximate_fields']) == {'rate_u_per_min', 'interval_total_units'}
    with core.store.tx() as db:
        envelope, _ = core._artifact(db, core._run(db, run['id'], 'alice'), 'policy')
        assert envelope['payload'] == slot['data']
        assert envelope['id'] == slot['artifact_id']


@pytest.mark.parametrize('value,unit', [(3.0000000000000004, 'U/h'), (0.05000000000000001, 'U/min')])
def test_conversion_does_not_round_unsafe_rate_into_boundary(setup, value, unit):
    core, run = setup
    set_rate(core, value, unit)
    assert core.execute_models('alice', run['id'])['reason'] == 'ENGINEERING_SAFETY_REJECTED'
    assert count(core, 'drafts') == count(core, 'releases') == 0


@pytest.mark.parametrize('value,unit', [(-0.6, 'U/h'), (float('nan'), 'U/h'),
    (float('inf'), 'U/h'), (True, 'U/h'), ('0.6', 'U/h'), (61, 'U/h'),
    (0.01, 'U'), (0.01, 'mg/h'), (0.01, None)])
def test_invalid_or_ambiguous_units_and_values_never_convert(setup, value, unit):
    core, run = setup
    set_rate(core, value, unit)
    with pytest.raises(GateError, match='INVALID_MODEL_OUTPUT'):
        core.execute_models('alice', run['id'])
    assert count(core, 'artifacts') == 1 and count(core, 'drafts') == 0


@pytest.mark.parametrize('fault', ['source_unit', 'source_value', 'fraction', 'configuration', 'revocation', 'missing_version'])
def test_conversion_evidence_and_configuration_revalidated_on_read(setup, fault):
    core, run = setup
    released = release(core, run)
    if fault == 'revocation':
        core.revoke_version(CONVERSION_VERSION)
    else:
        with core.store.tx() as db:
            if fault == 'missing_version':
                row = db.execute('SELECT versions FROM runs').fetchone()
                versions = json.loads(row['versions'])
                versions.remove(CONVERSION_VERSION)
                db.execute('UPDATE runs SET versions=?', (canonical(versions),))
            elif fault == 'configuration':
                row = db.execute('SELECT body FROM run_profiles').fetchone()
                body = json.loads(row['body'])
                body['unit_conversion']['arithmetic'] = 'round_before_check'
                db.execute('UPDATE run_profiles SET body=?,digest=?', (canonical(body), digest(body)))
            else:
                row = db.execute('SELECT body FROM drafts').fetchone()
                body = json.loads(row['body'])
                conversion = body['sections'][1]['action_display']['conversion']
                if fault == 'fraction': conversion['exact']['rate_u_per_min']['numerator'] = '99'
                else: conversion[fault] = 'U/min' if fault == 'source_unit' else 999
                db.execute('UPDATE drafts SET body=?,digest=?', (canonical(body), digest(body)))
    with pytest.raises(GateError): core.get_release('alice', released['release_id'])


def test_legacy_profile_and_report_are_not_silently_upgraded(setup):
    core, run = setup
    # Recreate the pre-v2 frozen run before any jobs, then execute its original registry.
    with core.store.tx() as db:
        row = db.execute('SELECT body FROM run_profiles').fetchone()
        profile = dict(legacy_fixture_profile(), binding=json.loads(row['body'])['binding'])
        versions = [LEGACY_PROFILE_VERSION, profile['rule_version'], profile['template_version'],
                    profile['prediction']['version'], profile['policy']['version'], 'technical-notice-v1']
        versions += [executor.version for name, executor in core._fixtures.items() if name in ('medical', 'ethics')]
        db.execute('UPDATE runs SET versions=?', (canonical(versions),))
        db.execute('UPDATE run_profiles SET body=?,digest=?', (canonical(profile), digest(profile)))
    released = release(core, run)
    report = core.get_release('alice', released['release_id'])['report']
    assert report['template_version'] == 'basal-fixture-template-v1'
    slot = report['sections'][1]
    assert slot['action_display']['rate_u_per_hour'] == '0.60'
    assert 'conversion' not in slot['action_display']
    raw = slot['data']['action']
    with pytest.raises(ValidationError): BasalAction.model_validate(dict(raw, action_unit='U/h'))
    with pytest.raises(ValidationError): BasalAction.model_validate(dict(raw, action_value=0.0000001))
