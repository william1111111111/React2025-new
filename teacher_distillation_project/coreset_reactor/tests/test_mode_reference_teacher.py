import numpy as np
import torch
from coreset_reactor.train_latent_reference_teacher import latent_ccc
from coreset_reactor.train_mode_reference_teacher import entropy_matched_log_prior,mode_latent_ccc


def test_prior_normalizes_and_matches_entropy():
    rng=np.random.default_rng(6)
    cost=rng.uniform(.01,10,size=(9,100))
    cost[:,0]=0
    logp=entropy_matched_log_prior(cost,8)
    np.testing.assert_allclose(np.exp(logp).sum(1),1,atol=1e-10)
    np.testing.assert_allclose(np.exp(-(np.exp(logp)*logp).sum(1)),8,atol=1e-8)
    assert (np.argmax(logp,axis=1)==0).all()


def test_degenerate_prior_does_not_break_identical_modes():
    logp=entropy_matched_log_prior(np.zeros((1,12)),8)
    np.testing.assert_allclose(np.exp(logp),np.full((1,12),1/12))


def test_uniform_prior_is_exact_uniform_reference_ablation():
    x=torch.tensor([[.1,.4,.6],[.2,.3,.4]],requires_grad=True)
    logp=torch.full_like(x,-np.log(3))
    a=mode_latent_ccc(x,logp)
    b=latent_ccc(x)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.autograd.grad(a.sum(),x,retain_graph=True)[0],
                               torch.autograd.grad(b.sum(),x)[0])


def test_prior_point_mass_recovers_fixed_reference():
    x=torch.tensor([[.1,.4,.6]],requires_grad=True)
    logp=torch.tensor([[-float('inf'),0.,-float('inf')]])
    value=mode_latent_ccc(x,logp)
    torch.testing.assert_close(value,x[:,1])
    torch.testing.assert_close(torch.autograd.grad(value.sum(),x)[0],torch.tensor([[0.,1.,0.]]))
