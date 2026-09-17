"""Causal, patient-isolated retrospective Loop samples. No model fitting or interpolation."""
from pathlib import Path
from collections import OrderedDict
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '预处理_v1'
STEP_SECONDS = 300
HISTORY = 72
DURATIONS = (5, 15, 30, 60)
CORE = ['cgm_mmol_l', 'basal_u_prev_5min', 'bolus_recorded_u_prev_5min']
AUX = ['carbs_recorded_g_prev_5min', 'exercise_event_count_prev_5min']
GAMMA_HOUR = 0.9  # Research convention per elapsed hour, not the paper's per-slot 0.9.


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def columns(mode='core'):
    if mode not in ('core', 'retrospective_multimodal'):
        raise ValueError('Unknown feature mode')
    return CORE + (AUX if mode == 'retrospective_multimodal' else [])


def feature_names(mode='core'):
    names = columns(mode)
    return (names + [n + '__observed' for n in names]
            + [n + '__age_log_hours' for n in names]
            + [n + '__age_known' for n in names]
            + ['elapsed_grid_hours', 'relative_to_decision_hours'])


def raw_features(frame, mode='core'):
    """No carry-forward values. Ages are causal metadata, not imputed measurements."""
    names = columns(mode)
    values = frame[names].to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    observed[:, 0] &= frame.cgm_mask.to_numpy(bool)
    if mode == 'retrospective_multimodal':
        observed[:, 3] &= ~frame.carbs_ambiguous_prev_5min.to_numpy(bool)
        # Zero event count means no exercise record, not observed lack of exercise.
        observed[:, 4] &= values[:, 4] > 0
    values[~observed] = np.nan
    n = len(frame)
    t = frame.timestamp_utc.astype('int64').to_numpy(dtype=np.float64) / 1e9
    age = np.zeros_like(values)
    age_known = np.zeros_like(observed)
    for k in range(len(names)):
        event_time = t.copy()
        if k == 0:
            event_time = pd.to_datetime(frame.cgm_observed_time).astype('int64').to_numpy(dtype=np.float64) / 1e9
        last = np.maximum.accumulate(np.where(observed[:, k], event_time, -np.inf))
        known = np.isfinite(last)
        age_known[:, k] = known
        age[known, k] = np.log1p(np.minimum(np.maximum(t[known] - last[known], 0), 86400) / 3600)
    return values, observed, age, age_known


def validate_frame(frame, patient_id, split):
    if set(frame.patient_id) != {patient_id} or set(frame['split']) != {split}:
        raise ValueError('Patient or split mismatch')
    times = frame.timestamp_utc.astype('int64').to_numpy()
    if not np.all(np.diff(times) == STEP_SECONDS * 10**9):
        raise ValueError('Nonunique or discontinuous grid: do not drop missing rows')
    good = frame.cgm_mask.to_numpy(bool)
    obs = pd.to_datetime(frame.cgm_observed_time).astype('int64').to_numpy()
    if not (np.all(np.isfinite(frame.cgm_mmol_l.to_numpy()[good]))
            and np.all(obs[good] <= times[good])
            and np.all(times[good] - obs[good] <= STEP_SECONDS * 10**9)):
        raise ValueError('CGM has an invalid value, future timestamp or stale source')


def duration_steps(minutes):
    if minutes not in DURATIONS:
        raise ValueError(f'Duration must be one of {DURATIONS}')
    return minutes // 5


def eligible_rows(frame, minutes=5):
    """Only real constant-rate blocks; never average differing actions into one."""
    k = duration_steps(minutes)
    n = len(frame)
    if n <= k:
        return np.empty(0, dtype=np.int64)
    start = np.arange(n - k)
    valid = np.ones(len(start), dtype=bool)
    flags = frame.rl_transition_eligible.to_numpy(bool)
    rate = frame.basal_action_u_h.to_numpy(float)
    ep = frame.episode_id.to_numpy()
    for j in range(k):
        valid &= flags[start + j] & (ep[start + j] == ep[start])
        valid &= np.isfinite(rate[start + j]) & (rate[start + j] == rate[start])
    valid &= start >= HISTORY - 1
    valid &= frame.cgm_mask.to_numpy(bool)[start + k]
    return start[valid]


def paper_status_score(glucose_mmol_l):
    """RL-DITR author glu2risk implementation, using mg/dL inside log().

    Author commit 5080fdbe, ts/datasets/rl.py: multiplier outside the subtraction.
    The printed Methods formula groups the multiplier differently; do not copy that grouping.
    """
    g = np.asarray(glucose_mmol_l, dtype=np.float64)
    if not np.all(np.isfinite(g) & (g > 0)):
        raise ValueError('Reward requires an actual positive glucose observation')
    b = g * 18.0
    risk = 10 * (1.509 * (np.log(b)**1.084 - 5.381))**2
    return np.where(b < 70.0, -1.0, 1.0 - np.clip(risk, 0, 15.5) / 7.75)


def interval_reward(next_glucose):
    scores = paper_status_score(next_glucose)
    # Rewards and discount use real elapsed time so regrouping does not change the return.
    weights = GAMMA_HOUR ** (np.arange(len(scores)) / 12)
    return float(np.sum(scores * weights) / 12)


def encode_history(raw, end, normalizer, mode='core'):
    if end < HISTORY - 1:
        raise ValueError('Insufficient history')
    names = columns(mode)
    sl = slice(end - HISTORY + 1, end + 1)
    values, observed, age, age_known = [a[sl].copy() for a in raw]
    mu = np.array([normalizer[n]['mean'] for n in names])
    sd = np.array([normalizer[n]['scale'] for n in names])
    # Zero is only a masked, normalized tensor placeholder. Files retain NaN.
    values = np.where(observed, (values - mu) / sd, 0)
    elapsed = np.full((HISTORY, 1), 1 / 12)
    relative = np.arange(-HISTORY + 1, 1).reshape(-1, 1) / 12
    result = np.concatenate([values, observed, age, age_known, elapsed, relative], axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite encoded input')
    return result.astype(np.float32)


def history_union(n, starts):
    difference = np.zeros(n + 1, dtype=np.int64)
    np.add.at(difference, starts - HISTORY + 1, 1)
    np.add.at(difference, starts + 1, -1)
    return np.cumsum(difference[:-1]) > 0


def prepare():
    """Verify frozen inputs, write compact indices and train-only normalization."""
    out = ROOT / 'prepared'
    out.mkdir(exist_ok=True)
    (out / 'indices').mkdir(exist_ok=True)
    manifest_path = DATA / '审计/rl_patient_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    if manifest.patient_id.duplicated().any():
        raise ValueError('Patient appears in more than one split')
    contract_path = ROOT / 'contract.json'
    contract_hash = sha(contract_path)
    indices = {(s, m): [] for s in manifest['split'].unique() for m in DURATIONS}
    moments = {n: [0, 0., 0.] for n in CORE + AUX}
    report = {'manifest_sha256': sha(manifest_path), 'contract_sha256': contract_hash,
              'source_hashes_verified': 0, 'patients_disjoint': True,
              'counts': {}, 'source_files': {}, 'normalizer_fit_split': 'train',
              'normalizer_patient_ids': manifest.loc[manifest['split'].eq('train'), 'patient_id'].tolist()}
    for record in manifest.to_dict('records'):
        pid, split = record['patient_id'], record['split']
        path = DATA / 'RL数据集' / split / (pid + '.parquet')
        digest = sha(path)
        if digest != record['file_sha256']:
            raise ValueError(f'Source hash changed: {pid}')
        frame = pd.read_parquet(path)
        validate_frame(frame, pid, split)
        report['source_hashes_verified'] += 1
        report['source_files'][pid] = {'split': split, 'sha256': digest}
        starts5 = eligible_rows(frame)
        if len(starts5) != int(record['rl_transitions']):
            raise ValueError(f'Frozen qualification differs: {pid}')
        for minutes in DURATIONS:
            starts = eligible_rows(frame, minutes)
            # IDs and row positions only: overlapping 6h windows are not copied to disk.
            indices[(split, minutes)].append(pd.DataFrame({'patient_id': pid, 'row_index': starts}))
        if split == 'train':
            keep = history_union(len(frame), starts5)
            values, observed, _, _ = raw_features(frame, 'retrospective_multimodal')
            for j, name in enumerate(CORE + AUX):
                v = values[keep & observed[:, j], j]
                moments[name][0] += len(v)
                moments[name][1] += float(v.sum())
                moments[name][2] += float(np.dot(v, v))
        if report['source_hashes_verified'] % 25 == 0:
            print('prepared patients', report['source_hashes_verified'], flush=True)
    normalization = {}
    for name, (count, total, squared) in moments.items():
        mean = total / count if count else 0.
        scale = np.sqrt(max(squared / count - mean * mean, 0.)) if count else 1.
        normalization[name] = {'mean': mean, 'scale': float(scale) if scale > 1e-12 else 1.,
                               'observed_training_rows': count}
    (out / 'normalization.json').write_text(json.dumps(normalization, indent=2) + '\n')
    for (split, minutes), pieces in indices.items():
        idx = pd.concat(pieces, ignore_index=True)
        idx.to_parquet(out / 'indices' / f'{split}_{minutes}min.parquet', index=False)
        report['counts'][f'{split}_{minutes}min'] = {'patients': int(idx.patient_id.nunique()), 'samples': len(idx)}
    report['artifact_hashes'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                               if p.is_file() and p.name != 'audit.json'}
    report['clinical_ready'] = False
    report['models_trained'] = False
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['counts'], indent=2))


class LoopDataset:
    """Map-style dataset returning NumPy arrays; compatible with PyTorch DataLoader.

    Exactly one patient cached per worker. No torch dependency for preparation.
    A source is hashed the first time that worker opens it.
    """
    def __init__(self, split='train', duration_minutes=5, mode='core', allow_sealed_test=False):
        duration_steps(duration_minutes)
        if split == 'sealed_test' and not allow_sealed_test:
            raise ValueError('Sealed test requires explicit final-evaluation access')
        self.split, self.minutes, self.mode = split, duration_minutes, mode
        self.names = feature_names(mode)
        out = ROOT / 'prepared'
        self.audit = json.loads((out / 'audit.json').read_text())
        if sha(ROOT / 'contract.json') != self.audit['contract_sha256']:
            raise ValueError('Contract changed: rebuild and revalidate')
        for rel in ['normalization.json', f'indices/{split}_{duration_minutes}min.parquet']:
            if sha(out / rel) != self.audit['artifact_hashes'][rel]:
                raise ValueError(f'Prepared artifact changed: {rel}')
        self.index = pd.read_parquet(out / 'indices' / f'{split}_{duration_minutes}min.parquet')
        self.normalizer = json.loads((out / 'normalization.json').read_text())
        self.cache = OrderedDict()
        self.verified = set()

    def __len__(self):
        return len(self.index)

    def _patient(self, pid):
        if pid not in self.cache:
            record = self.audit['source_files'][pid]
            if record['split'] != self.split:
                raise ValueError('Index crosses split')
            path = DATA / 'RL数据集' / self.split / (pid + '.parquet')
            if pid not in self.verified:
                if sha(path) != record['sha256']:
                    raise ValueError('Patient file changed')
                self.verified.add(pid)
            frame = pd.read_parquet(path)
            validate_frame(frame, pid, self.split)
            raw = raw_features(frame, self.mode)
            starts = eligible_rows(frame, self.minutes)
            allowed = np.zeros(len(frame), bool)
            allowed[starts] = True
            self.cache.clear()
            self.cache[pid] = (frame, raw, allowed)
        return self.cache[pid]

    def __getitem__(self, item):
        record = self.index.iloc[int(item)]
        pid, i = record.patient_id, int(record.row_index)
        frame, raw, allowed = self._patient(pid)
        k = duration_steps(self.minutes)
        if i >= len(allowed) or not allowed[i]:
            raise ValueError('Index points to unqualified target')
        j = i + k
        glucose = frame.cgm_mmol_l.iloc[i+1:j+1].to_numpy(float)
        basal = frame.basal_u_next_5min.iloc[i:j].to_numpy(float)
        action = float(frame.basal_action_u_h.iloc[i])
        if not np.isclose(basal.sum(), action * self.minutes / 60, atol=1e-7, rtol=1e-7):
            raise ValueError('Dose-duration inconsistency')
        # End of usable logged actions is truncation, never evidence of a clinical terminal.
        continuation = j < len(allowed) and allowed[j] and frame.episode_id.iloc[j] == frame.episode_id.iloc[i]
        next_state_valid = bool(frame.cgm_mask.iloc[j] and frame.basal_bolus_history_6h_ok.iloc[j]
                                and frame.cgm_history_6h_gap_le1h.iloc[j] and frame.in_study_window.iloc[j])
        state = encode_history(raw, i, self.normalizer, self.mode)
        next_state = encode_history(raw, j, self.normalizer, self.mode)
        # Cointerventions are outcomes/context for patient-model diagnostics, NOT actor inputs.
        bolus = frame.bolus_recorded_u_next_5min.iloc[i:j].to_numpy(float)
        return {'patient_id': pid, 'split': self.split, 'row_index': i,
                'episode_id': int(frame.episode_id.iloc[i]),
                'timestamp_ns': int(frame.timestamp_utc.iloc[i].value),
                'state': state, 'next_state': next_state,
                'action_u_h': np.float32(action), 'action_duration_minutes': np.float32(self.minutes),
                'basal_interval_u': np.float32(basal.sum()),
                'reward': np.float32(interval_reward(glucose)),
                'discount': np.float32(GAMMA_HOUR ** (self.minutes / 60)),
                'terminated': False, 'truncated': not continuation,
                'bootstrap_valid': next_state_valid,
                'end_reason': 'continues' if continuation else 'data_eligibility_boundary',
                'target_glucose_mmol_l': glucose.astype(np.float32),
                'target_wtr': ((glucose >= 3.9) & (glucose <= 10)).astype(np.float32),
                'outcome_bolus_logged_u': np.nan_to_num(bolus).astype(np.float32),
                'outcome_bolus_record_present': np.isfinite(bolus),
                'outcome_bolus_actual_absence_verified': np.zeros(k, dtype=bool)}

    def trajectory(self, item, max_steps=12):
        """Variable length, no padding or invented terminal return. Each future step stays in episode."""
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        first = self[item]
        samples = [first]
        pid = first['patient_id']
        rows = self.index[self.index.patient_id.eq(pid)]
        positions = dict(zip(rows.row_index, rows.index))
        current = first
        while len(samples) < max_steps and not current['truncated']:
            next_row = current['row_index'] + duration_steps(self.minutes)
            current = self[positions[next_row]]
            samples.append(current)
        return samples

    def iter_batches(self, batch_size=32, seed=20260914):
        """Shuffle patients and their samples while avoiding random patient-file I/O per sample."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        rng = np.random.default_rng(seed)
        groups = self.index.groupby('patient_id', sort=True).indices
        patients = list(groups)
        rng.shuffle(patients)
        for pid in patients:
            positions = groups[pid].copy()
            rng.shuffle(positions)
            for start in range(0, len(positions), batch_size):
                batch = [self[int(i)] for i in positions[start:start+batch_size]]
                yield {name: ([s[name] for s in batch] if isinstance(batch[0][name], str)
                              else np.stack([s[name] for s in batch])) for name in batch[0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'smoke'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        ds = LoopDataset()
        batch = next(ds.iter_batches())
        print({'samples': len(ds), 'features': ds.names,
               'batch_state_shape': batch['state'].shape,
               'finite': bool(np.isfinite(batch['state']).all())})
