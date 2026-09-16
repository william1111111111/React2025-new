import unittest
import torch
from .model import Judge
class Contracts(unittest.TestCase):
 def test_padding_and_order(self):
  torch.manual_seed(123);torch.set_num_threads(2);m=Judge().eval();x=torch.randn(2,32,851);y=torch.randn(2,32,25);t=torch.arange(32)[None].expand(2,-1).float()/30;mask=torch.ones(2,32,dtype=torch.bool)
  with torch.no_grad():
   a=m(x,y,t,t,mask,mask);pad=lambda z:torch.nn.functional.pad(z,(0,0,0,8)) if z.ndim==3 else torch.nn.functional.pad(z,(0,8));b=m(pad(x),pad(y),pad(t),pad(t),pad(mask),pad(mask));c=m(x.flip(0),y.flip(0),t,t,mask,mask).flip(0)
  torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5);torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-5)
 def test_generated_bank_shape(self):
  import tempfile
  import numpy as np
  from pathlib import Path
  from .data import load
  with tempfile.TemporaryDirectory() as d:
   for name,shape,expected in [('bank',(16,32,25),(16,32,25)),('coeff',(32,1,58),(32,58))]:
    p=Path(d)/(name+'.npy');np.save(p,np.zeros(shape,np.float32));self.assertEqual(load(str(p)).shape,expected)
 def test_unknown(self):
  d=torch.tensor([1.,-2.],requires_grad=True);w=torch.zeros_like(d);loss=(torch.nn.functional.softplus(-d)*w).sum();loss.backward();self.assertTrue(torch.equal(d.grad,torch.zeros_like(d)))
 def test_source_sensitivity_and_grad(self):
  torch.manual_seed(123);torch.set_num_threads(2);m=Judge().eval();x=torch.randn(1,32,851);y=torch.randn(1,32,25);t=torch.arange(32)[None].float()/30;mask=torch.ones(1,32,dtype=torch.bool);a=m(x,y,t,t,mask,mask);b=m(x*0,y,t,t,mask,mask);self.assertGreater(float((a-b).abs().max()),1e-6);a.sum().backward();self.assertGreater(float(m.x.local[0].weight.grad.norm()),0)
if __name__=='__main__':unittest.main()
