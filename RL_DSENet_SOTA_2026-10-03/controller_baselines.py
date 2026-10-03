"""Observable-only research baselines; these are not official reproductions.

Zone/IOB/ROC motivation: Deshpande et al., TCST 2023, doi:10.1109/TCST.2023.3291573.
IOB and predictive reduction/recovery mechanics: OpenAPS oref0 commit
88cf032aa74ff25f69464a7d9cd601ee3940c0b3, lib/determine-basal/determine-basal.js.
The finite-history PI(D) formula and all numeric gains below are this project's
explicit research assumptions, not those authors' fitted controller parameters.

ISF is estimated as 1800/(24*anchor/basal_fraction), with basal_fraction=0.5.
This is an empirical prior, not a hidden simulator parameter or patient-specific
measurement. The carbohydrate glucose effect 3.6 mg/dL per gram is likewise a
fixed heuristic (1800/500 rule ratio), and the COB proxy is unvalidated.
Only basal rate is returned. No bolus, CR, CF, patient ID, future meal or true BG
is supplied. Bounds match the new 0..min(20,2*anchor) research action contract.
"""
from dataclasses import asdict, dataclass
import numpy as np
from physiologic_features import physiologic_state, zone_error


@dataclass(frozen=True)
class ControllerParameters:
    zone_lower: float = 100.
    zone_upper: float = 140.
    proportional_gain: float = 0.6
    integral_gain: float = 0.2
    derivative_minutes: float = 10.
    correction_hours: float = 1.5
    iob_correction_hours: float = 2.
    basal_fraction: float = 0.5
    isf_rule_numerator: float = 1800.
    carb_glucose_factor: float = 3.6
    suspend_cgm: float = 70.
    suspend_prediction: float = 75.
    reduction_prediction: float = 100.
    prediction_minutes: float = 20.
    maximum_anchor_ratio: float = 2.
    maximum_rate_u_h: float = 20.


def action_upper_bound(anchor):
    return np.minimum(20., 2*np.asarray(anchor, dtype=np.float64))


def low_glucose_cap(state, proposed, params):
    """Recomputed each call: reduction ends when observed risk abates."""
    g = state['cgm_mg_dl']
    trend = np.minimum(state['roc15'], state['roc30'])
    minimum = np.minimum(g, g + params.prediction_minutes * trend)
    fraction = np.clip((minimum - params.suspend_prediction) /
                       (params.reduction_prediction - params.suspend_prediction), 0, 1)
    result = np.where(minimum < params.reduction_prediction,
                      np.minimum(proposed, state['anchor_u_h'] * fraction), proposed)
    return np.where((g <= params.suspend_cgm) | (minimum <= params.suspend_prediction), 0., result)


class IOBZoneController:
    """Finite-window zone PI with filtered trend and recorded-insulin feedback.

    Zone error of the 60 min mean is an integral proxy; recomputing from history
    avoids an unobservable state and indefinite integrator windup. Above-basal
    delivered insulin subtracts from later corrections; basal deficits can
    support recovery. Recorded bolus not covered by the crude recorded-COB
    estimate limits additional insulin. This is a baseline to test, not proof
    that unrecorded food or insulin are absent.
    """
    def __init__(self, params=None):
        self.params = params or ControllerParameters()
        p = self.params
        if not (0 < p.maximum_anchor_ratio <= 2 and 0 < p.maximum_rate_u_h <= 20):
            raise ValueError('Controller parameters cannot widen the frozen action bounds')
        if not (p.zone_lower < p.zone_upper and 0 < p.basal_fraction < 1
                and p.correction_hours > 0 and p.iob_correction_hours > 0
                and p.isf_rule_numerator > 0 and p.suspend_prediction < p.reduction_prediction):
            raise ValueError('Invalid research controller parameters')
        self.parameters = asdict(self.params)

    def action(self, history, anchor):
        s = physiologic_state(history, anchor)
        if np.any(s['last_cgm_age_min'] > 10) or np.any(s['basal_fraction300'] < .95):
            raise ValueError('Research controller requires recent CGM and recorded basal history')
        p = self.params
        a = s['anchor_u_h']
        isf = p.isf_rule_numerator / (24*a/p.basal_fraction)
        error = zone_error(s['cgm_mg_dl'], p.zone_lower, p.zone_upper)
        integral_error = zone_error(s['mean_cgm60'], p.zone_lower, p.zone_upper)
        correction = (p.proportional_gain * error + p.integral_gain * integral_error
                      + p.derivative_minutes * np.clip(s['roc30'], -3, 3)) / (isf*p.correction_hours)
        carb_covered_u = s['recorded_cob_g'] * p.carb_glucose_factor / isf
        unopposed_bolus = np.maximum(s['bolus_iob_u'] - carb_covered_u, 0)
        outstanding = s['net_basal_iob_u'] + unopposed_bolus
        proposed = a + correction - outstanding / p.iob_correction_hours
        upper = np.minimum(p.maximum_rate_u_h, p.maximum_anchor_ratio*a)
        proposed = np.clip(proposed, 0, upper)
        return np.clip(low_glucose_cap(s, proposed, p), 0, upper).astype(np.float32)


class HoldController:
    def action(self, history, anchor):
        # Validate the same observed-input contract as the other controllers.
        s = physiologic_state(history, anchor)
        return np.minimum(s['anchor_u_h'], action_upper_bound(anchor)).astype(np.float32)


class ExplorationController:
    """Rule/random mixture for training coverage, with 30--90 min random holds.

    RNG state is an experiment stream, never a patient identifier. A controller
    instance owns a fixed batch of episode lanes; make a new instance on reset.
    Low-glucose override can interrupt any random hold, and is recomputed every
    five minutes. This limiter does not make the generated episodes safe.
    """
    def __init__(self, seed=260915, rule_probability=.5):
        if not 0 <= rule_probability <= 1:
            raise ValueError('rule_probability must be in [0,1]')
        self.rng = np.random.default_rng(seed)
        self.rule_probability = rule_probability
        self.rule = IOBZoneController()
        self.remaining = None
        self.ratio = None
        self.use_rule = None

    def action(self, history, anchor):
        s = physiologic_state(history, anchor)
        n = len(s['anchor_u_h'])
        if self.remaining is None:
            self.remaining = np.zeros(n, dtype=int)
            self.ratio = np.zeros(n)
            self.use_rule = np.zeros(n, dtype=bool)
        if len(self.remaining) != n:
            raise ValueError('Exploration batch lanes cannot change within an episode')
        renew = self.remaining <= 0
        count = int(renew.sum())
        self.remaining[renew] = self.rng.integers(6, 19, size=count)
        self.ratio[renew] = self.rng.uniform(0, 2, size=count)
        self.use_rule[renew] = self.rng.random(count) < self.rule_probability
        rule_action = self.rule.action(history, anchor)
        proposed = np.where(self.use_rule, rule_action, self.ratio * s['anchor_u_h'])
        self.remaining -= 1
        guarded = low_glucose_cap(s, proposed, self.rule.params)
        return np.clip(guarded, 0, action_upper_bound(anchor)).astype(np.float32)


def make_controller(name, seed=260915):
    if name == 'physiology':
        return IOBZoneController()
    if name == 'explore':
        return ExplorationController(seed=seed)
    if name == 'hold':
        return HoldController()
    raise ValueError('Unknown controller: ' + name)
