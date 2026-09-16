import unittest,sys
import numpy as np
import torch
from tslearn.metrics import dtw
from .quality import active_dtw,ccc,hinges
from .rewards import weights
class QualityTests(unittest.TestCase):
 def test_active_forward_and_gradient(self):
  torch.manual_seed(23)
  for n,m in [(7,5),(8,8),(3,6)]:
   x=torch.randn(n,3,dtype=torch.double,requires_grad=True);y=torch.randn(m,3,dtype=torch.double);v,ref=active_dtw(x,y);self.assertAlmostEqual(float(v),ref,places=10);self.assertAlmostEqual(float(v),dtw(x.detach().numpy(),y.numpy()),places=10)
   g=torch.autograd.grad(v,x)[0];eps=1e-5;xp=x.detach().clone();xm=x.detach().clone();xp[1,1]+=eps;xm[1,1]-=eps;fd=(active_dtw(xp,y)[0]-active_dtw(xm,y)[0])/(2*eps);self.assertAlmostEqual(float(g[1,1]),float(fd),places=5)
 def test_zero_ties_and_padding(self):
  for n,m in [(4,4),(4,6)]:
   x=torch.zeros(n,2,dtype=torch.double,requires_grad=True);y=torch.zeros(m,2,dtype=torch.double);v,_=active_dtw(x,y);g=torch.autograd.grad(v,x)[0];assert torch.isfinite(g).all() and (g==0).all()
  x=torch.randn(9,2,dtype=torch.double,requires_grad=True);y=torch.randn(5,2,dtype=torch.double);v,_=active_dtw(x[:4],y);g=torch.autograd.grad(v,x)[0];assert (g[4:]==0).all()
 def test_ccc_native_constants(self):
  sys.path.insert(0,'/home/zhengshiyi/react2025');from framework.metrics.FRC import concordance_correlation_coefficient as official
  for constant in [False,True]:
   x=torch.randn(33,25,requires_grad=True);y=torch.randn(33,25)
   if constant:y[:,0]=1.;y[:,1]=0.
   v=ccc(x,y);ref=official(y.numpy(),x.detach().numpy())[0];assert abs(float(v)-ref)<1e-6;g=torch.autograd.grad(v,x)[0];assert torch.isfinite(g).all()
  x=torch.zeros(8,25,requires_grad=True);v=ccc(x,torch.zeros_like(x));g=torch.autograd.grad(v,x)[0];assert torch.isfinite(g).all()
 def test_absolute_feedback_and_separate_hinges(self):
  cal={'Z_Q':1.,'Z_B':.1};q=np.arange(10)*.01-.2
  for arm in ['Q0-quality-guarded','Q1-coverage-guarded']:
   r=weights(q,np.ones(10),np.ones(10,dtype=bool),arm,cal,False);assert (r<.5).all();assert (weights(q-.5,np.ones(10),np.ones(10,dtype=bool),arm,cal,False)<r).all()
  c=torch.tensor(-1.);d=torch.tensor(-5.);a,b=hinges(c,d,torch.tensor(0.),torch.tensor(0.),{'c':1,'d':1});assert a>0 and b==0
if __name__=='__main__':unittest.main()
