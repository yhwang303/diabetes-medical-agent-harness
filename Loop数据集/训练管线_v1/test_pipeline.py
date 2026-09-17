import unittest
import numpy as np
import pandas as pd
from pipeline import (CORE, AUX, HISTORY, GAMMA_HOUR, raw_features, encode_history,
                      eligible_rows, validate_frame, paper_status_score, interval_reward,
                      history_union, LoopDataset)


def frame(n=120):
    t = pd.date_range('2020-01-01', periods=n, freq='5min')
    d = pd.DataFrame({'patient_id': 'LOOP_000001', 'split': 'train', 'timestamp_utc': t,
                      'cgm_observed_time': t, 'cgm_mmol_l': 6., 'cgm_mask': True,
                      'basal_u_prev_5min': .1, 'bolus_recorded_u_prev_5min': np.nan,
                      'carbs_recorded_g_prev_5min': np.nan, 'exercise_event_count_prev_5min': 0,
                      'carbs_ambiguous_prev_5min': False, 'rl_transition_eligible': False,
                      'basal_action_u_h': 1.2, 'episode_id': 1})
    d.loc[72:n-2, 'rl_transition_eligible'] = True
    return d


NORMAL = {k: {'mean': 0., 'scale': 1.} for k in CORE + AUX}


class PipelineTests(unittest.TestCase):
    def test_real_zero_distinct_from_missing(self):
        d = frame()
        d.loc[80, 'bolus_recorded_u_prev_5min'] = 0.
        x = encode_history(raw_features(d), 80, NORMAL)
        self.assertEqual(x[-1, 2], 0)
        self.assertEqual(x[-1, 5], 1)
        self.assertEqual(x[-2, 2], 0)
        self.assertEqual(x[-2, 5], 0)

    def test_missing_cgm_no_fill(self):
        d = frame()
        d.loc[79, ['cgm_mmol_l', 'cgm_mask']] = [np.nan, False]
        x = encode_history(raw_features(d), 80, NORMAL)
        self.assertEqual(x[-2, 0], 0)
        self.assertEqual(x[-2, 3], 0)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isnan(d.cgm_mmol_l.iloc[79]))

    def test_future_mutation_does_not_change_current_state(self):
        d = frame()
        expected = encode_history(raw_features(d), 80, NORMAL)
        d.loc[81:, CORE] = 999.
        actual = encode_history(raw_features(d), 80, NORMAL)
        np.testing.assert_array_equal(expected, actual)

    def test_no_long_duration_average(self):
        d = frame()
        self.assertIn(80, eligible_rows(d, 30))
        d.loc[82, 'basal_action_u_h'] = 2.4
        self.assertNotIn(80, eligible_rows(d, 30))
        self.assertIn(80, eligible_rows(d, 5))

    def test_gap_and_episode_boundary(self):
        d = frame()
        d.loc[82, 'rl_transition_eligible'] = False
        self.assertNotIn(80, eligible_rows(d, 30))
        d.loc[82, 'rl_transition_eligible'] = True
        d.loc[82:, 'episode_id'] = 2
        self.assertNotIn(80, eligible_rows(d, 30))

    def test_cross_patient_and_split_rejected(self):
        d = frame()
        d.loc[0, 'patient_id'] = 'LOOP_000002'
        with self.assertRaises(ValueError):
            validate_frame(d, 'LOOP_000001', 'train')
        d = frame()
        d.loc[0, 'split'] = 'validation'
        with self.assertRaises(ValueError):
            validate_frame(d, 'LOOP_000001', 'train')

    def test_dropped_grid_and_future_observation_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(frame().drop(index=5), 'LOOP_000001', 'train')
        d = frame()
        d.loc[4, 'cgm_observed_time'] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            validate_frame(d, 'LOOP_000001', 'train')

    def test_reward_units_and_invalid_values(self):
        self.assertEqual(float(paper_status_score(60/18)), -1)
        b = 120.
        expected = 1 - min(15.5, 10 * (1.509 * (np.log(b)**1.084 - 5.381))**2) / 7.75
        self.assertAlmostEqual(float(paper_status_score(b/18)), expected)
        self.assertGreater(float(paper_status_score(112.5/18)), .99)
        for bad in [0., np.nan, -1.]:
            with self.assertRaises(ValueError):
                paper_status_score(bad)

    def test_reward_duration_composition(self):
        x = np.linspace(4., 10., 12)
        self.assertAlmostEqual(interval_reward(x), interval_reward(x[:6]) +
                               GAMMA_HOUR**.5 * interval_reward(x[6:]))

    def test_normalization_only_observed_and_history_union(self):
        used = history_union(150, np.array([80, 81]))
        self.assertEqual(np.flatnonzero(used).tolist(), list(range(9, 82)))
        d = frame()
        norm = {k: {'mean': 100., 'scale': 2.} for k in CORE}
        x = encode_history(raw_features(d), 80, norm)
        self.assertEqual(x[-1, 2], 0)  # Missing is not standardized as a real zero.

    def test_unknown_exercise_and_conflicting_carbs_masked(self):
        d = frame()
        d.loc[80, 'carbs_recorded_g_prev_5min'] = 50.
        d.loc[80, 'carbs_ambiguous_prev_5min'] = True
        values, masks, _, _ = raw_features(d, 'retrospective_multimodal')
        self.assertFalse(masks[80, 3])
        self.assertFalse(masks[80, 4])

    def test_sealed_test_requires_explicit_access(self):
        with self.assertRaises(ValueError):
            LoopDataset('sealed_test')

    def test_truncation_is_not_terminal_and_preserves_bootstrap(self):
        d = frame()
        d['basal_u_next_5min'] = .1
        d['bolus_recorded_u_next_5min'] = np.nan
        d['basal_bolus_history_6h_ok'] = True
        d['cgm_history_6h_gap_le1h'] = True
        d['in_study_window'] = True
        d.loc[81:, 'rl_transition_eligible'] = False
        ds = LoopDataset.__new__(LoopDataset)
        ds.minutes, ds.mode, ds.split = 5, 'core', 'train'
        ds.normalizer = NORMAL
        ds.index = pd.DataFrame({'patient_id': ['LOOP_000001'], 'row_index': [80]})
        allowed = np.zeros(len(d), bool)
        allowed[eligible_rows(d)] = True
        ds.cache = {'LOOP_000001': (d, raw_features(d), allowed)}
        sample = ds[0]
        self.assertTrue(sample['truncated'])
        self.assertFalse(sample['terminated'])
        self.assertTrue(sample['bootstrap_valid'])
        self.assertEqual(sample['state'].shape, (72, 14))
        self.assertFalse(sample['outcome_bolus_record_present'][0])
        self.assertFalse(sample['outcome_bolus_actual_absence_verified'][0])

    def test_multistep_reward_uses_each_observation_not_just_endpoint(self):
        a = np.array([6., 6., 6.])
        b = np.array([3., 6., 6.])
        self.assertLess(interval_reward(b), interval_reward(a))


if __name__ == '__main__':
    unittest.main()
