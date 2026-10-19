import numpy as np
import torch

from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.train_latent_reference_teacher import reference_ccc,latent_ccc,LatentReferenceLoss
from regnn.conditional_model import ConditionalREGNNLoss,masked_ccc


def test_ccc_matches_vectorized_official_on_real_shapes_and_constants():
    rng=np.random.default_rng(913)
    p=rng.normal(size=(3,70,25)).astype('float32')
    y=rng.normal(size=(5,70,25)).astype('float32')
    p[:,:,0]=0
    y[:,:,1]=0
    got=reference_ccc(torch.from_numpy(p),torch.from_numpy(y)).max(-1).values.numpy()
    np.testing.assert_allclose(got,candidate_frc(y,p),atol=1e-6,rtol=1e-5)


def test_soft_reference_aggregation_bounds_and_duplication_invariance():
    scores=torch.tensor([[.1,.2,.3],[-.3,-.4,-.2]],requires_grad=True)
    value=latent_ccc(scores)
    assert (value<=scores.max(-1).values).all()
    assert (value>=scores.mean(-1)).all()
    torch.testing.assert_close(value,latent_ccc(scores.repeat(1,2)))
    value.sum().backward()
    torch.testing.assert_close(scores.grad.sum(-1),torch.ones(2))
    posterior=torch.softmax(scores.detach()/.05,dim=-1)
    torch.testing.assert_close(scores.grad,posterior)
    kl=(posterior*(posterior.log()+np.log(scores.shape[-1]))).sum(-1)
    variational=(posterior*scores.detach()).sum(-1)-.05*kl
    torch.testing.assert_close(value.detach(),variational)


def test_single_reference_reduces_to_baseline_loss_and_gradient():
    torch.manual_seed(66)
    target=torch.rand(1,30,25)
    prediction=torch.rand(1,30,25,requires_grad=True)
    class Bank:
        def at(self,index,length,device): return target.to(device)
    mask=torch.ones(1,30,dtype=torch.bool)
    velocity=torch.zeros(1,dtype=torch.bool)
    baseline=ConditionalREGNNLoss()(prediction,target,mask,velocity)
    packed=torch.cat((target,torch.zeros(1,30,1)),dim=-1)
    result=LatentReferenceLoss(bank=Bank(),audit_every=0)(prediction,packed,mask,velocity)
    torch.testing.assert_close(result['loss'],baseline['loss'],atol=1e-6,rtol=1e-6)
    g1=torch.autograd.grad(result['loss'],prediction,retain_graph=True)[0]
    g2=torch.autograd.grad(baseline['loss'],prediction)[0]
    torch.testing.assert_close(g1,g2,atol=1e-6,rtol=1e-5)


def test_padding_never_enters_reference_matching():
    torch.manual_seed(8)
    prediction=torch.rand(2,25,25,requires_grad=True)
    target=torch.rand(2,25,25)
    refs=torch.rand(7,25,25)
    class Bank:
        def at(self,index,length,device): return refs[:,:length].to(device)
    mask=torch.arange(25)[None]<torch.tensor([17,25])[:,None]
    packed=torch.cat((target,torch.zeros(2,25,1)),dim=-1)
    loss=LatentReferenceLoss(bank=Bank(),audit_every=0)
    first=loss(prediction,packed,mask)['loss']
    changed=prediction.clone()
    changed[~mask]=1000
    packed[~mask]=1000
    torch.testing.assert_close(first,loss(changed,packed,mask)['loss'])


def test_constant_prediction_keeps_ccc_learning_gradient():
    p=torch.full((1,20,25),.5,requires_grad=True)
    y=torch.rand(1,20,25,generator=torch.Generator().manual_seed(1))
    expected=masked_ccc(p,y,torch.ones(1,20,dtype=torch.bool)).mean()
    actual=reference_ccc(p,y).mean()
    torch.testing.assert_close(actual,expected)
    g1=torch.autograd.grad(expected,p,retain_graph=True)[0]
    g2=torch.autograd.grad(actual,p)[0]
    assert g1.norm()>0
    torch.testing.assert_close(g1,g2,atol=1e-6,rtol=1e-5)
