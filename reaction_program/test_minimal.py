import unittest
import torch
from .motion_codec import MotionCodec,CodecConfig
from .reaction_program import ProgramBatch,require_observed_label
from .motion_generator import OracleProgramExecutor

def program(n=17):
 return ProgramBatch(torch.tensor([[0,1]]),torch.tensor([[True,True]]),torch.tensor([[[1.,5.,12.],[4.,10.,float(n)]]]),torch.tensor([[.5,.5]]),torch.tensor([n]))
def data(t=17):
 y=torch.cat([torch.randint(0,2,(1,t,15)).float(),torch.rand(1,t,2)*2-1,torch.rand(1,t,8).softmax(-1)],-1);return y
class MinimalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_codec_tail_mask_and_gradient(self):
  torch.manual_seed(123);m=MotionCodec();y=data();length=torch.tensor([13]);out=m(y,length)
  self.assertEqual(out['tokens'].shape,(1,5,2));self.assertTrue((out['tokens'][:,4]==-1).all());self.assertTrue((out['reconstruction'][:,13:]==0).all())
  self.assertTrue(torch.allclose(out['reconstruction'][:,:13,17:].sum(-1),torch.ones(1,13),atol=1e-6))
  changed=y.clone();changed[:,13:]=999;self.assertTrue(torch.equal(out['tokens'],m(changed,length)['tokens']))
  m.loss(y,length,out)['total'].backward()
  for module in [m.encoder,m.decoder,m.quantizer]:self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
 def test_token_decode_roundtrip(self):
  torch.manual_seed(123);m=MotionCodec().eval();y=data();n=torch.tensor([17]);o=m(y,n)
  torch.testing.assert_close(m.decode_tokens(o['tokens'],n,17),o['reconstruction'],atol=1e-6,rtol=1e-5)
 def test_executor_program_and_clock_sensitivity(self):
  torch.manual_seed(123);m=OracleProgramExecutor(['TEST_ACTION_A','TEST_ACTION_B']);p=program();t=torch.full((1,5,2),512)
  a,_=m(t,p);q=program();q.action_ids[:]=1;b,_=m(t,q)
  self.assertGreater(float((a-b).abs().max()),1e-4)
  r=program();r.timing[0,0]+=1;c,_=m(t,r);self.assertGreater(float((a-c).abs().max()),1e-4)
  loss=m.loss(torch.zeros_like(t),p,torch.zeros(t.shape));loss.backward();self.assertGreater(float(m.action.weight.grad.abs().sum()),0)
 def test_determinism_kprefix(self):
  torch.manual_seed(123);m=OracleProgramExecutor(['TEST_ACTION_A','TEST_ACTION_B']).eval();u=torch.rand(1,3,3,5,2);p=program()
  a=m.generate(p,u);self.assertTrue(torch.equal(a,m.generate(p,u)));self.assertTrue(torch.equal(a[:,:2],m.generate(p,u[:,:2])))
 def test_unknown_not_maintain(self):
  p=program();p.action_mask[:]=False
  with self.assertRaises(ValueError):p.validate(2)
  with self.assertRaises(ValueError):require_observed_label({'split':'train','status':'pending_observation','actions':None})
 def test_bad_clock(self):
  p=program();p.timing[0,0,0]=-1
  with self.assertRaises(ValueError):p.validate(2)
if __name__=='__main__':unittest.main()
