import importlib.util,unittest
from pathlib import Path
import numpy as np,pandas as pd
s=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('02_build.py'));b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class AlignmentTests(unittest.TestCase):
 def test_equal_rate_overlap_is_union_not_sum(self):
  seg=b.segments(np.array([0.,150.]),np.array([300.,450.]),np.array([1.2,1.2]),np.array([False,False]));np.testing.assert_allclose(b.integrals(seg,np.array([0.]),'dose'),[.1]);np.testing.assert_allclose(b.integrals(seg,np.array([0.]),'known'),[300.])
 def test_conflicting_rate_excludes_overlap(self):
  seg=b.segments(np.array([0.,150.]),np.array([300.,450.]),np.array([1.,2.]),np.array([False,False]));np.testing.assert_allclose(b.integrals(seg,np.array([0.]),'rate_conflict'),[150.]);np.testing.assert_allclose(b.integrals(seg,np.array([0.]),'known'),[150.])
 def test_same_time_distinct_patients_never_share(self):
  a=b.segments(np.array([0.]),np.array([300.]),np.array([1.2]),np.array([False]));c=b.segments(np.array([0.]),np.array([300.]),np.array([2.4]),np.array([False]));self.assertAlmostEqual(b.integrals(a,np.array([0.]),'dose')[0],.1);self.assertAlmostEqual(b.integrals(c,np.array([0.]),'dose')[0],.2)
 def test_gap_has_no_delivered_coverage(self):
  seg=b.segments(np.array([0.,200.]),np.array([100.,300.]),np.array([1.,1.]),np.array([False,False]));self.assertEqual(b.integrals(seg,np.array([0.]),'known')[0],200.)
 def test_suspend_zero_is_observed_delivery(self):
  seg=b.segments(np.array([0.]),np.array([300.]),np.array([0.]),np.array([False]));self.assertEqual(b.integrals(seg,np.array([0.]),'dose')[0],0.);self.assertEqual(b.integrals(seg,np.array([0.]),'known')[0],300.)
 def test_invalid_event_blocks_known_interval(self):
  seg=b.segments(np.array([0.,100.]),np.array([300.,200.]),np.array([1.,np.nan]),np.array([False,True]));self.assertEqual(b.integrals(seg,np.array([0.]),'invalid_event')[0],100.)
 def test_millisecond_boundaries_no_tolerance_fill(self):
  seg=b.segments(np.array([0.,150.001]),np.array([150.,300.]),np.array([1.,1.]),np.array([False,False]));self.assertAlmostEqual(b.integrals(seg,np.array([0.]),'known')[0],299.999)
 def test_exact_change_at_boundary(self):
  seg=b.segments(np.array([0.,300.]),np.array([300.,600.]),np.array([1.2,2.4]),np.array([False,False]));np.testing.assert_allclose(b.integrals(seg,np.array([0.,300.]),'dose'),[.1,.2])
 def test_random_sweep_against_independent_midpoint_reference(self):
  rng=np.random.default_rng(20260914)
  for j in range(30):
   starts=rng.integers(0,500,20).astype(float);ends=starts+rng.integers(1,400,20);rates=rng.choice([0.,.5,1.,2.],20);invalid=rng.random(20)<.1;seg=b.segments(starts,ends,rates,invalid)
   for x in seg.itertuples():
    mid=(x.start_s+x.end_s)/2;active=(starts<=mid)&(ends>mid);r=np.unique(rates[active]);expected='invalid_event' if (invalid&active).any() else ('gap' if not active.any() else ('known' if len(r)==1 else 'rate_conflict'));self.assertEqual(x.status,expected)
    if expected=='known':self.assertEqual(x.rate_u_h,r[0])
 def test_cgm_bin_assignment_cannot_look_ahead(self):
  ts=np.array([299.,300.,301.,899.]);grid=np.ceil(ts/300)*300;self.assertTrue((grid>=ts).all());self.assertTrue(((grid-ts)<300).all());np.testing.assert_equal(grid,[300.,300.,600.,900.])
 def test_history_excludes_current_future_interval(self):
  x=np.ones(100,bool);x[72]=False;y=b.rolling_all(x);self.assertTrue(y[72]);self.assertFalse(y[73]);self.assertFalse(y[:72].any())
 def test_missing_dose_not_assumed_present(self):
  d=pd.DataFrame({'event_time':pd.to_datetime([150.],unit='s'),'dose':[np.nan]});counts,values=b.bin_events(d,np.array([0.,300.,600.]),'dose',True);self.assertEqual(counts[1],1);self.assertTrue(np.isnan(values).all())
if __name__=='__main__':unittest.main(verbosity=2)
