import itertools
import numpy as np
import torch
from .policy import Policy,normal_log_prob,replace_low_frequency
from .rewards import coverage,utility,difference_advantages
from mode_supervision.plans import basis


def test_identity_prefix_and_tail():
    torch.manual_seed(123);torch.set_num_threads(2);p=Policy().double();c=torch.randn(1,256,dtype=torch.float64)
    for n in (1,3,751):
        q=basis(torch.arange(n,dtype=torch.float64)*.04+11);noise=torch.randn(3,n,24,dtype=torch.float64)
        out,z,m=replace_low_frequency(noise,q,p,c);assert torch.equal(out,noise)
        prefix,_,_=replace_low_frequency(noise[:2],q,p,c);assert torch.equal(prefix,out[:2])
        assert torch.allclose(p.log_prob(z,c.expand(3,-1),m),normal_log_prob(z,m),atol=1e-12)


def test_inverse_density_and_gradients():
    torch.manual_seed(3);p=Policy().double();c=torch.randn(2,256,dtype=torch.float64);z0=torch.randn(2,96,dtype=torch.float64);m=torch.ones_like(z0,dtype=torch.bool);m[:,72:]=False
    with torch.no_grad():
        for l in p.layers:l.net[-1].weight.normal_(0,.01)
    z,lq=p.rsample_and_log_prob(z0,c,m);x,ild=p.inverse(z,c,m)
    assert torch.allclose(x,z0*m,atol=1e-10) and torch.allclose(lq,p.log_prob(z,c,m),atol=1e-10)
    assert (z[~m]==0).all()
    detached=z.detach();loss=-(p.log_prob(detached,c,m)*torch.tensor([1.,2.])).mean();loss.backward()
    assert sum(v.grad.abs().sum() for v in p.parameters())>0 and detached.grad_fn is None
    # Numerical Jacobian on a small valid subspace checks the claimed determinant.
    mm=torch.zeros(1,96,dtype=torch.bool);mm[:,:3]=True;small=torch.randn(3,dtype=torch.float64,requires_grad=True)
    def f(v):return p.transform(torch.nn.functional.pad(v,(0,93))[None],c[:1],mm)[0][0,:3]
    jac=torch.autograd.functional.jacobian(f,small);_,ld=p.transform(torch.nn.functional.pad(small,(0,93))[None],c[:1],mm)
    assert torch.allclose(torch.linalg.slogdet(jac)[1],ld[0],atol=1e-9)


def test_coverage():
    a=np.array([[1.,0.]])
    assert coverage(a)==coverage(np.repeat(a,2,0))
    assert coverage(np.r_[a,[[0.,1.]]])>coverage(a)
    u=utility(np.array([[1.,0.]]),np.array([[10.,0.]]),np.array([True]),np.zeros((1,2)),np.zeros((2,2)),np.ones(2),.5,1.)
    assert not u.any() # favorable CCC and DTW from different targets do not count


def test_exact_discrete_score_gradient():
    p=.3;full=diff=0.;K=3
    for seq in itertools.product((0,1),repeat=K):
        z=np.array(seq);prob=np.prod(np.where(z,p,1-p));score=z-p;u=np.stack([1-z,z],1).astype(float)
        adv=difference_advantages('R-coverage',np.zeros(K),np.zeros(K),np.zeros(K),u,u,{'phi':np.ones(2),'c':1,'d':1},[0,0,0])
        full+=prob*coverage(u)*score.sum();diff+=prob*np.dot(adv,score)
    assert abs(full-diff)<1e-12 and abs(diff)>1e-4
