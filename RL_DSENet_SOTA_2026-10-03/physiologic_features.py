"""Observable-history physiology estimates for simulation research, not patient care.

Input is the frozen Loop multimodal 72x22 tensor: five standardized values,
five observed masks, five record ages, five age-known masks, and two clocks.
Only observed values are decoded. A masked bolus/carbohydrate contributes no
*recorded* event; this never establishes that a real person received none.

Insulin residual/activity kernels use the exponential curve in OpenAPS oref0:
https://github.com/openaps/oref0/blob/88cf032aa74ff25f69464a7d9cd601ee3940c0b3/lib/iob/calculate.js#L123
The chosen 300 min duration and 75 min peak are fixed research assumptions.
Basal and bolus refer to amounts delivered during the previous five minutes;
their midpoint age is used. Net basal IOB is relative to the observed warmup
anchor, so it may be negative; it is not total insulin in the body.

Recorded COB is a deliberately simple three-hour linear remaining-carbohydrate
proxy, not measured absorption or a reproduction of OpenAPS COB. All features
are estimates from the last six hours, with record masks/ages kept explicit.

The insulin-curve equations are adapted from MIT-licensed OpenAPS code.
Copyright (c) 2015-2019 OpenAPS Contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np


NORMALIZER_PATH = Path(__file__).resolve().parent.parent / 'Loop数据集/训练管线_v2/prepared/normalization.json'
VALUE_NAMES = ('cgm_mmol_l', 'basal_u_prev_5min', 'bolus_recorded_u_prev_5min',
               'carbs_recorded_g_prev_5min', 'exercise_event_count_prev_5min')
INSULIN_DURATION_MIN = 300.
INSULIN_PEAK_MIN = 75.
CARB_DURATION_MIN = 180.
FEATURE_NAMES = (
    'cgm_centered_100', 'roc15_over3', 'roc30_over3', 'roc60_over3',
    'mean_cgm60_centered_100', 'min_cgm60_centered_100', 'std_cgm60_over50',
    'last_basal_over_anchor_minus1', 'mean_basal60_over_anchor_minus1',
    'recorded_bolus_iob_over_3h_anchor', 'net_basal_iob_over_3h_anchor',
    'recorded_bolus_activity_over_anchor_per_min', 'net_basal_activity_over_anchor_per_min',
    'recorded_cob_over60', 'recorded_carbs60_over60', 'recorded_bolus60_over_3h_anchor',
    'last_cgm_age_over30', 'cgm_fraction60', 'basal_fraction300',
    'bolus_record_fraction300', 'carb_record_fraction300',
    'last_positive_bolus_record_age_over360', 'last_positive_carb_record_age_over360',
    'anchor_over2', 'positive_bolus_record_in_history', 'positive_carb_record_in_history',
    'mean_zone_error60_over100', 'current_cgm_observed',
)
FEATURE_DIM = len(FEATURE_NAMES)


@lru_cache(maxsize=1)
def normalization():
    raw = NORMALIZER_PATH.read_bytes()
    values = json.loads(raw)
    return (np.array([values[n]['mean'] for n in VALUE_NAMES]),
            np.array([values[n]['scale'] for n in VALUE_NAMES]),
            hashlib.sha256(raw).hexdigest())


def insulin_kernel(age_min, duration_min=INSULIN_DURATION_MIN, peak_min=INSULIN_PEAK_MIN):
    """Return remaining fraction and activity fraction/min for a unit dose."""
    age = np.asarray(age_min, dtype=np.float64)
    if not 0 < peak_min < duration_min / 2 or np.any(age < 0):
        raise ValueError('Require nonnegative ages and 0 < peak < duration/2')
    tau = peak_min * (1 - peak_min / duration_min) / (1 - 2 * peak_min / duration_min)
    a = 2 * tau / duration_min
    scale = 1 / (1 - a + (1 + a) * np.exp(-duration_min / tau))
    t = np.minimum(age, duration_min)
    decay = np.exp(-t / tau)
    activity = scale / tau**2 * t * (1 - t / duration_min) * decay
    remaining = 1 - scale * (1 - a) * ((t**2 / (tau * duration_min * (1 - a))
                                      - t / tau - 1) * decay + 1)
    return (np.where(age < duration_min, np.clip(remaining, 0, 1), 0.),
            np.where(age < duration_min, np.maximum(activity, 0), 0.))


def decode_history(history, anchor):
    """Decode physical values and masks; masked placeholders are never read."""
    h = np.asarray(history, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (72, 22) or anchor.shape != (len(h),):
        raise ValueError('Expected history[N,72,22] and anchor[N]')
    if not np.all(np.isfinite(anchor)) or np.any(anchor <= 0):
        raise ValueError('Anchor must be a positive finite observed basal rate')
    mask_values = h[:, :, 5:10]
    if not np.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError('Observed masks must be binary')
    mask = mask_values.astype(bool)
    if not np.all(np.isfinite(h[:, :, :5][mask])):
        raise ValueError('Observed values must be finite')
    mean, scale, _ = normalization()
    raw = np.where(mask, h[:, :, :5] * scale + mean, 0.)
    raw[:, :, 0] *= 18
    # Float32 normalization can decode a true zero to tiny negative roundoff.
    if np.any(raw[:, :, 1:][mask[:, :, 1:]] < -1e-5):
        raise ValueError('Observed delivered doses/events cannot be negative')
    raw[:, :, 1:] = np.maximum(raw[:, :, 1:], 0)
    return raw, mask, anchor


def _mean(values, mask):
    return np.sum(np.where(mask, values, 0), axis=1) / np.maximum(mask.sum(1), 1)


def _slope(cgm, mask, bins):
    """Least-squares causal slope, including both ends of the named interval."""
    y, m = cgm[:, -bins:], mask[:, -bins:]
    x = np.broadcast_to(np.arange(bins) * 5., y.shape)
    xc = x - _mean(x, m)[:, None]
    yc = y - _mean(y, m)[:, None]
    numerator = np.sum(np.where(m, xc * yc, 0), axis=1)
    denominator = np.sum(np.where(m, xc**2, 0), axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def zone_error(glucose, lower=100., upper=140.):
    return np.minimum(glucose - lower, 0) + np.maximum(glucose - upper, 0)


def physiologic_state(history, anchor):
    """Physical-unit estimates; no patient identifier or simulator imports."""
    raw, mask, anchor = decode_history(history, anchor)
    n = len(raw)
    positions = np.arange(72)[None, :]
    age = (71 - np.arange(72)) * 5.
    dose_age = age + 2.5
    remaining, activity = insulin_kernel(dose_age)
    cgm, cmask = raw[:, :, 0], mask[:, :, 0]
    latest = np.max(np.where(cmask, positions, -1), axis=1)
    latest_basal = np.max(np.where(mask[:, :, 1], positions, -1), axis=1)
    last = cgm[np.arange(n), np.maximum(latest, 0)]
    # Zero-valued unavailable physical fields are paired with explicit masks/ages.
    last = np.where(latest >= 0, last, 0)
    basal = raw[:, :, 1]
    bolus = raw[:, :, 2]
    carbs = raw[:, :, 3]
    net_basal = np.where(mask[:, :, 1], basal - anchor[:, None] / 12, 0)
    mean60 = _mean(cgm[:, -12:], cmask[:, -12:])
    min60 = np.min(np.where(cmask[:, -12:], cgm[:, -12:], np.inf), axis=1)
    min60 = np.where(np.isfinite(min60), min60, 0)
    std60 = np.sqrt(_mean((cgm[:, -12:] - mean60[:, None])**2, cmask[:, -12:]))
    bolus_event = mask[:, :, 2] & (bolus > 1e-6)
    carb_event = mask[:, :, 3] & (carbs > 1e-6)
    bolus_age = np.min(np.where(bolus_event, dose_age, 360.), axis=1)
    carb_age = np.min(np.where(carb_event, dose_age, 360.), axis=1)
    last_basal = basal[np.arange(n), np.maximum(latest_basal, 0)] * 12
    return dict(
        cgm_mg_dl=last, roc15=_slope(cgm, cmask, 4), roc30=_slope(cgm, cmask, 7),
        roc60=_slope(cgm, cmask, 13), mean_cgm60=mean60, min_cgm60=min60, std_cgm60=std60,
        last_basal_u_h=np.where(latest_basal >= 0, last_basal, 0),
        mean_basal60_u_h=_mean(basal[:, -12:], mask[:, -12:, 1]) * 12,
        bolus_iob_u=bolus @ remaining, net_basal_iob_u=net_basal @ remaining,
        bolus_activity_u_min=bolus @ activity, net_basal_activity_u_min=net_basal @ activity,
        recorded_cob_g=carbs @ np.maximum(1 - dose_age / CARB_DURATION_MIN, 0),
        recorded_carbs60_g=carbs[:, -12:].sum(1), recorded_bolus60_u=bolus[:, -12:].sum(1),
        last_cgm_age_min=np.where(latest >= 0, (71 - latest) * 5., 360.),
        cgm_fraction60=cmask[:, -12:].mean(1), basal_fraction300=mask[:, -60:, 1].mean(1),
        bolus_record_fraction300=mask[:, -60:, 2].mean(1),
        carb_record_fraction300=mask[:, -60:, 3].mean(1),
        last_bolus_record_age_min=bolus_age, last_carb_record_age_min=carb_age,
        bolus_record_present=bolus_event.any(1), carb_record_present=carb_event.any(1),
        mean_zone_error60=_mean(zone_error(cgm[:, -12:]), cmask[:, -12:]),
        current_cgm_observed=cmask[:, -1], anchor_u_h=anchor,
    )


def features(history, anchor):
    """Return 28 finite dimensionless features, in FEATURE_NAMES order."""
    s = physiologic_state(history, anchor)
    a = s['anchor_u_h']
    values = (
        (s['cgm_mg_dl'] - 120) / 100, s['roc15'] / 3, s['roc30'] / 3, s['roc60'] / 3,
        (s['mean_cgm60'] - 120) / 100, (s['min_cgm60'] - 120) / 100, s['std_cgm60'] / 50,
        s['last_basal_u_h'] / a - 1, s['mean_basal60_u_h'] / a - 1,
        s['bolus_iob_u'] / (3*a), s['net_basal_iob_u'] / (3*a),
        s['bolus_activity_u_min'] * 60 / a, s['net_basal_activity_u_min'] * 60 / a,
        s['recorded_cob_g'] / 60, s['recorded_carbs60_g'] / 60, s['recorded_bolus60_u'] / (3*a),
        s['last_cgm_age_min'] / 30, s['cgm_fraction60'], s['basal_fraction300'],
        s['bolus_record_fraction300'], s['carb_record_fraction300'],
        s['last_bolus_record_age_min'] / 360, s['last_carb_record_age_min'] / 360,
        a / 2, s['bolus_record_present'], s['carb_record_present'],
        s['mean_zone_error60'] / 100, s['current_cgm_observed'],
    )
    result = np.stack(values, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite physiology features')
    return result
