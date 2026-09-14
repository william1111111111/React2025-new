"""Deterministic checks of sign, noncircular support and missing evidence."""
import unittest
import numpy as np
from run import scan,peak,diagnostic,POLICY
class Contract(unittest.TestCase):
    def test_known_shift_and_nonzero_origin(self):
        rng=np.random.default_rng(5);t=10+np.arange(2000)*.04;a=rng.normal(size=len(t))
        # M(t) = A(t-0.12): visual movement is delayed, delta=-0.12.
        m=np.interp(t-.12,t,a)
        scores,n=scan(t,m,t,a,(10,89),POLICY['lag_seconds'])
        self.assertAlmostEqual(peak(scores,POLICY['lag_seconds'])['delta_s'],-.12)
        self.assertEqual(n,1950)
    def test_no_constant_or_nan_pass(self):
        t=np.arange(500)*.04
        for m in [np.zeros(500),np.full(500,np.nan)]:
            self.assertEqual(diagnostic(t,m,t,np.sin(t))['status'],'unresolved')
    def test_short_overlap(self):
        t=np.arange(10)*.04
        scores,n=scan(t,t,t,t,(0,.36),POLICY['lag_seconds'])
        self.assertTrue(all(x is None for x in scores));self.assertEqual(n,0)
if __name__=='__main__':unittest.main()
