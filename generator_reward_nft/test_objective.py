import unittest
import numpy as np
import torch
from .objective import core,nft
from .rewards import weights
class Contract(unittest.TestCase):
    def test_masked_core_sign_and_neutral(self):
        old=torch.randn(2,7,24,dtype=torch.double);target=torch.randn_like(old);mask=torch.arange(7)[None]<torch.tensor([3,7])[:,None]
        for r in [0.,.5,1.]:
            new=old.clone().requires_grad_();loss=core(new,old,target,new.new_full((2,),r),mask);g=torch.autograd.grad(loss,new)[0]
            expected=2*(2*r-1)*(old-target)/(mask.sum(1)[:,None,None]*24*2);expected.masked_fill_(~mask[...,None],0)
            torch.testing.assert_close(g,expected)
    def test_endpoint_neutral_and_padding(self):
        old=torch.randn(2,7,24,dtype=torch.double);u=torch.randn_like(old);clean=torch.randn_like(old);mask=torch.arange(7)[None]<torch.tensor([3,7])[:,None];tau=torch.tensor([.2,.7],dtype=torch.double);new=old.clone().requires_grad_()
        loss=nft(new,old,u,clean,tau,new.new_full((2,),.5),mask);g=torch.autograd.grad(loss,new)[0];torch.testing.assert_close(g,torch.zeros_like(g))
    def test_no_support_or_ties_do_not_get_positive_credit(self):
        cal={'alpha':10,'Z':{'N0-quality':1.,'N1-quality-coverage':1.}}
        for arm in cal['Z']:
            r=weights(np.ones(10),np.ones(10),np.ones(10,dtype=bool),arm,cal);np.testing.assert_array_equal(r,np.full(10,.5))
            r=weights(np.arange(10.),np.arange(10.),np.zeros(10,dtype=bool),arm,cal);assert (r<=.5).all()
if __name__=='__main__':unittest.main()
