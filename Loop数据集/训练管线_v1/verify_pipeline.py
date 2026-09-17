"""Independent readback checks on every train/validation patient; no test policy evaluation."""
import json
import time
import numpy as np
from pipeline import ROOT, CORE, DURATIONS, LoopDataset, raw_features, encode_history, sha


def main():
    result = {'sample_checks': 0, 'trajectory_checks': 0, 'future_mutation_checks': 0,
              'duration_splits_checked': [], 'failures': [], 'cuda_tested': False}
    started = time.perf_counter()
    for split in ['train', 'validation']:
        for minutes in DURATIONS:
            ds = LoopDataset(split, minutes)
            for pid, positions in ds.index.groupby('patient_id', sort=True).indices.items():
                for loc in np.unique([positions[0], positions[len(positions)//2], positions[-1]]):
                    s = ds[int(loc)]
                    d, _, _ = ds._patient(pid)
                    i, k = s['row_index'], minutes // 5
                    assert s['patient_id'] == pid and s['split'] == split
                    assert s['state'].shape == (72, 14) and np.isfinite(s['state']).all()
                    actual_next = d.cgm_mmol_l.iloc[i+1:i+k+1].to_numpy()
                    np.testing.assert_allclose(s['target_glucose_mmol_l'], actual_next, rtol=1e-6)
                    # Independently compute physical integral and published reward expression.
                    np.testing.assert_allclose(s['basal_interval_u'], d.basal_u_next_5min.iloc[i:i+k].sum(), rtol=1e-6)
                    b = actual_next * 18
                    score = np.where(b < 70, -1, 1-np.minimum(15.5, 10*(1.509*(np.log(b)**1.084-5.381))**2)/7.75)
                    expected = sum((.9**(j/12))*v/12 for j, v in enumerate(score))
                    np.testing.assert_allclose(s['reward'], expected, atol=1e-7, rtol=1e-6)
                    assert not s['terminated']
                    result['sample_checks'] += 1
                if minutes == 5:
                    item = int(positions[len(positions)//2])
                    s = ds[item]
                    d, _, _ = ds._patient(pid)
                    mutated = d.copy()
                    mutated.loc[mutated.index > s['row_index'], CORE] = 777.
                    x = encode_history(raw_features(mutated), s['row_index'], ds.normalizer)
                    np.testing.assert_array_equal(x, s['state'])
                    result['future_mutation_checks'] += 1
                    seq = ds.trajectory(item, max_steps=12)
                    assert len({v['patient_id'] for v in seq}) == 1
                    assert len({v['episode_id'] for v in seq}) == 1
                    assert all(seq[j+1]['row_index']-seq[j]['row_index'] == 1 for j in range(len(seq)-1))
                    result['trajectory_checks'] += 1
            result['duration_splits_checked'].append(f'{split}_{minutes}min')
            print('verified', split, minutes, result['sample_checks'], flush=True)
    ds = LoopDataset()
    batch = next(ds.iter_batches(batch_size=32))
    assert batch['state'].shape == (32, 72, 14)
    assert np.isfinite(batch['state']).all() and np.isfinite(batch['reward']).all()
    result['batch_shape'] = list(batch['state'].shape)
    result['seconds'] = time.perf_counter()-started
    result['prepared_audit_sha256'] = sha(ROOT/'prepared/audit.json')
    result['code_sha256'] = {p.name: sha(p) for p in ROOT.glob('*.py')}
    result['scope'] = 'Data loader correctness only; not policy efficacy or counterfactual validity'
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
