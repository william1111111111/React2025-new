import copy,torch
from reaction_flow.shared_noise import make_initial_noise,recording_bases
from reaction_flow.shared_model import upgrade
from reaction_flow.tests.test_flow import tiny
from reaction_flow.config import FlowConfig

def test_covariance_and_padding():
    torch.manual_seed(123);local=torch.randn(60000,8,1,dtype=torch.double);g=torch.randn(60000,1,dtype=torch.double);rho=.05
    e=make_initial_noise(local,g,rho)
    assert abs(float(e[:,0].var())-1)<.02
    assert abs(float((e[:,0]*e[:,1]).mean())-rho)<.015
    assert abs(float(e.mean(1).var())-(rho+(1-rho)/8))<.005
    cov=(1-rho)*torch.eye(8)+rho*torch.ones(8,8);assert torch.linalg.eigvalsh(cov).min()>0
    assert torch.equal(make_initial_noise(local,None,0),local)
    mask=torch.arange(8)[None]<torch.tensor([3,5])[:,None]
    result=make_initial_noise(local[:2],g[:2],rho,mask);assert not result[~mask].any()

def test_recording_key_prefix_and_blocks():
    cfg=FlowConfig();a,g=recording_bases(cfg,'fixed-source',1600,4);b,h=recording_bases(cfg,'fixed-source',750,2)
    assert torch.equal(a[:2,:750],b) and torch.equal(g[:2],h)
    assert not torch.equal(g[0],g[1]) and not torch.equal(a[:,:750],a[:,750:1500])
    e=make_initial_noise(a,g,.05);remainder=e-(.95**.5)*a
    torch.testing.assert_close(remainder[:,0],remainder[:,1500],atol=3e-7,rtol=1e-5)

def test_public_sampler_compatibility_prefix_chunk():
    model=tiny().double().eval();shared=upgrade(copy.deepcopy(model),0.)
    x=[torch.randn(1,12,c,dtype=torch.double) for c in (768,25,58)];n=torch.tensor([9]);z=torch.randn(1,4,12,24,dtype=torch.double);g=torch.randn(1,4,24,dtype=torch.double)
    old=model.sample(*x,n,4,z,integration_steps=2);new=shared.sample(*x,n,4,z,integration_steps=2,global_noise=g)
    torch.testing.assert_close(old,new,atol=0,rtol=0)
    shared.rho=.05;y=shared.sample(*x,n,4,z,integration_steps=2,global_noise=g)
    chunk=shared.sample(*x,n,4,z,integration_steps=2,global_noise=g,candidate_chunk=1)
    prefix=shared.sample(*x,n,2,z[:,:2],integration_steps=2,global_noise=g[:,:2])
    torch.testing.assert_close(y,chunk,atol=1e-12,rtol=1e-12);torch.testing.assert_close(y[:,:2],prefix,atol=1e-12,rtol=1e-12)
    assert not y[:,:,9:].any()
