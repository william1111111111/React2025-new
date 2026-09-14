import unittest
import numpy as np
from refine import torch,gain_for,interval_iou,coverage,F,vad
class Contract(unittest.TestCase):
 def test_gain_no_clipping_or_original_mutation(self):
  x=torch.tensor([0.,.02,-.01]);copy=x.clone();g=gain_for(x)
  self.assertLessEqual(float((x*g).abs().max()),.950001);self.assertLessEqual(g,1000);self.assertTrue(torch.equal(x,copy))
 def test_silence_stays_silent(self):
  x=torch.zeros(16000);self.assertEqual(float((x*gain_for(x)).abs().sum()),0)
 def test_interval_origins(self):
  a=[{'start':100,'end':200}];b=[{'start':150,'end':250}];self.assertAlmostEqual(interval_iou(a,b),1/3);self.assertIsNone(interval_iou([],[]))
 def test_context_coverage(self):
  self.assertAlmostEqual(coverage(10,12,[{'start_s':11,'end_s':13}]),.5)
if __name__=='__main__':unittest.main()
