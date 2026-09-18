"""Keep all frozen origins; distinguish CGM labels from complete action prefixes."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'Loop数据集/训练管线_v2'))
import pipeline

def main():
    summary = {}
    identities = []
    for split in ('train', 'validation'):
        ds = pipeline.LoopDataset(split=split, mode='retrospective_multimodal')
        out = ROOT / 'forecast_masks' / split
        out.mkdir(parents=True, exist_ok=True)
        records = []
        identities.append(set(ds.index.patient_id))
        for pid, index in ds.index.groupby('patient_id'):
            frame, raw, allowed = ds._patient(pid)
            starts = index.row_index.to_numpy(dtype='int64')
            # RL episode_id encodes action eligibility, not CGM continuity.
            # Reuse the existing >1h-CGM-gap segment contract for factual labels.
            episodes = frame.cgm_segment_id.to_numpy(dtype='int32')
            study = frame.in_study_window.to_numpy(bool)
            # Include the existing last in-study transition's next-state CGM.
            observed = raw[1][:, 0] & (study | np.r_[False, study[:-1]])
            rows = starts[:, None] + np.arange(1, 49)[None]
            clipped = np.minimum(rows, len(frame) - 1)
            mask = (rows < len(frame)) & (episodes[clipped] == episodes[starts, None]) & observed[clipped]
            assert (episodes[starts] > 0).all() and mask[:, 0].all()
            path = out / (pid + '.npz')
            np.savez_compressed(path, starts=starts, episode=episodes, observed=observed)
            records.append({'patient': pid, 'origins': len(starts),
                            'labels_per_horizon': mask.sum(0).tolist(),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        summary[split] = {'patients': len(records), 'origins': sum(r['origins'] for r in records),
                          'labels_per_horizon': np.sum([r['labels_per_horizon'] for r in records], axis=0).tolist(),
                          'records': records,
                          'segment_definition': 'existing CGM segment: split on observed gap >1h; not RL action-eligibility episode',
                          'all_origins_have_first_label': True}
        print(split, summary[split]['patients'], summary[split]['origins'], flush=True)
    assert not identities[0] & identities[1]
    assert summary['train']['origins'] == 1653421
    assert summary['validation']['origins'] == 424866
    (ROOT / 'forecast_masks/manifest.json').write_text(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
