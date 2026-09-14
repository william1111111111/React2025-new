import unittest
import numpy as np
from run import decide,windows
class Contract(unittest.TestCase):
 def test_single_and_two_voices(self):
  a=np.tile([1.,0.,0.],(8,1));self.assertEqual(decide(a,[1.5]*8)['count'],1)
  a[4:]=[0,1,0];r=decide(a,[1.5]*8);self.assertEqual(r['count'],2);self.assertEqual(r['labels'],[0]*4+[1]*4)
 def test_insufficient_not_single(self):
  self.assertIsNone(decide(np.tile([1.,0.],(3,1)),[1.5]*3)['count'])
 def test_single_outlier_not_two(self):
  a=np.tile([1.,0.],(8,1));a[-1]=[0,1];self.assertIsNone(decide(a,[1.5]*8)['count'])
 def test_short_and_nonoverlap(self):
  ws,short=windows([{'start':0,'end':8000},{'start':16000,'end':96000}])
  self.assertEqual(len(short),1);self.assertTrue(all(.8<=w['end_s']-w['start_s']<=2 for w in ws))
  self.assertTrue(all(a['end_s']<=b['start_s'] for a,b in zip(ws,ws[1:])))
if __name__=='__main__':unittest.main()
