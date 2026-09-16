import unittest
import numpy as np
import torch
from .temporal import nearest
from .sampling import EffectiveSampler
class Repairs(unittest.TestCase):
 def test_rounded_pts(self):
  for fps in [25,30]:
   p=np.round(7+np.arange(900)/fps,6);i,j,d=nearest(p,np.arange(200,600),4,1/fps);self.assertTrue(all(d['checks'].values()));self.assertTrue(np.all(j-i==4*fps))
 def test_support(self):
  p=np.arange(100)/30;i,j,d=nearest(p,np.arange(100),4,1/30);self.assertEqual(len(i),0);self.assertEqual(d['outside_support'],100)
 def test_missing_frame(self):
  p=np.delete(np.arange(1000)/30,400);i,j,d=nearest(p,np.arange(200,600),4,1/30);self.assertFalse(all(d['checks'].values()))
 def test_nearest_residual(self):
  p=np.arange(1000)/30;i,j,d=nearest(p,np.arange(100,500),4.01,1/30);self.assertLess(d['max_abs_residual_s'],.011)
 def test_unknown_and_split(self):
  rows=[dict(family='temporal',split=s,state=state,weight=w,source=f's{i}',group=f'g{i%2}') for i,(s,state,w) in enumerate([('RM_fit','PREFERRED',.5),('RM_fit','UNKNOWN',0),('RM_audit','PREFERRED',1),('RM_fit','PREFERRED',.5)])];ids=EffectiveSampler(rows).draw('temporal',8);self.assertEqual(set(ids),{0,3});x=torch.randn(2,requires_grad=True);(torch.nn.functional.softplus(-x)*0).sum().backward();self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))
if __name__=='__main__':unittest.main()
