import pytest
import torch
from .plans import basis, summarize, recording
from .models import PlanPrior, PlanInjection


def test_mean_irregular_and_origin():
    pts = torch.tensor([11., 11.04, 11.11, 11.16, 11.25], dtype=torch.float64)
    u = torch.randn(5,24, dtype=torch.float64, requires_grad=True)
    a,m = summarize(u,pts)
    assert torch.allclose(a[0],u.mean(0),atol=1e-12)
    assert torch.allclose(basis(pts),basis(pts+100),atol=1e-12)
    assert torch.allclose(basis(pts).T@basis(pts),torch.eye(4,dtype=torch.float64),atol=1e-12)
    a.square().sum().backward()
    assert u.grad.norm()>0

@pytest.mark.parametrize('n',[1,2,3,4,751])
def test_short_tail(n):
    u=torch.randn(n,24);pts=torch.arange(n,dtype=torch.float64)*.04
    a,m=recording(u,pts)
    assert a.shape==m.shape==((n+749)//750,96)
    assert int(m[-1].sum())==min(4,(n-1)%750+1)*24
    assert (a[~m]==0).all()


def test_invalid():
    for p in ([],[0,0],[2,1],[float('nan')]):
        with pytest.raises(ValueError):basis(p)


def test_detached_prior_zero_residual_and_sensitivity():
    torch.manual_seed(123);torch.set_num_threads(2)
    prior=PlanPrior();inject=PlanInjection();mask=torch.ones(2,3,96,dtype=torch.bool)
    context=torch.randn(2,3,256);pos=torch.randn(2,3,2);z=torch.randn(2,3,96)
    plan=prior.sample(z,context,pos,mask)
    assert not plan.requires_grad
    h=torch.randn(2,8,256);idx=torch.tensor([0,2])
    assert torch.equal(inject(h,plan,mask,idx),h)
    opt=torch.optim.AdamW(inject.parameters(),lr=.001)
    for _ in range(2):
        opt.zero_grad();inject(h,plan,mask,idx).square().mean().backward();opt.step()
    assert all(p.grad is None for p in prior.parameters())
    assert inject.project[0].weight.grad.norm()>0
    assert not torch.equal(inject(h,plan,mask,idx),inject(h,plan+1,mask,idx))
    assert torch.equal(prior.sample(z,context,pos,mask),plan)
