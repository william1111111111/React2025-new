import unittest
import numpy as np
from spectral import spectral
class Contract(unittest.TestCase):
 def test_one_allowed_not_forced_two(self):
  for p in [.2,.3,.4]:self.assertEqual(spectral(np.tile([1.,0],(8,1)),p)['count'],1)
 def test_two_disconnected_voice_groups(self):
  e=np.array([[1.,0]]*4+[[0.,1]]*4)
  for p in [.2,.3,.4]:
   r=spectral(e,p);self.assertEqual(r['count'],2);self.assertEqual(r['labels'],[0]*4+[1]*4)
 def test_insufficient_unknown(self):self.assertIsNone(spectral(np.array([[1.,0]]*3),.3)['count'])
if __name__=='__main__':unittest.main()
