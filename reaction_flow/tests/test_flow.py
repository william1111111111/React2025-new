import inspect
import torch
from reaction_flow.reaction_transform import ReactionTransform,contrast_matrix
from reaction_flow.flow_path import flow_path,flow_loss
from reaction_flow.sampler import ReactionFlow
from reaction_flow.config import FlowConfig
from hirp.config import HiRPConfig

def tiny():
    torch.manual_seed(4)
    return ReactionFlow(FlowConfig(d_model=16,blocks=2,heads=4,ff_width=32),HiRPConfig(d_model=16,nhead=4,num_layers=1,dim_feedforward=32,dropout=0.))

def test_transform_internal_and_boundary():
    torch.manual_seed(5);tr=ReactionTransform();q=contrast_matrix()
    torch.testing.assert_close(q.T@q,torch.eye(7,dtype=torch.double),atol=3e-16,rtol=0)
    torch.testing.assert_close(q.T@torch.ones(8,dtype=torch.double),torch.zeros(7,dtype=torch.double),atol=3e-16,rtol=0)
    y=torch.cat((torch.rand(3,19,15,dtype=torch.double)*.8+.1,torch.rand(3,19,2,dtype=torch.double)*1.6-.8,torch.randn(3,19,8,dtype=torch.double).softmax(-1)),-1)
    torch.testing.assert_close(tr.inverse(tr(y)),y,atol=1e-14,rtol=0)
    edge=torch.cat((torch.zeros(2,15,dtype=torch.double),torch.ones(2,2,dtype=torch.double),torch.eye(8,dtype=torch.double)[:2]),-1)
    back=tr.inverse(tr(edge));assert torch.isfinite(back).all();torch.testing.assert_close(back,tr.regularize(edge),atol=1e-14,rtol=0)
    assert float((edge-back).abs().max())<.00071

def test_path_mask_and_equal_occurrence_weight():
    y=torch.randn(2,7,24);e=torch.randn_like(y);mask=torch.arange(7)[None]<torch.tensor([3,7])[:,None];t=torch.tensor([0.,1.])
    state,target=flow_path(y,e,t,mask)
    torch.testing.assert_close(state[0,:3],e[0,:3]);torch.testing.assert_close(state[1],y[1]);assert not state[0,3:].any()
    prediction=target.clone();prediction[0,:3]+=1;prediction[0,3:]=100
    torch.testing.assert_close(flow_loss(prediction,target,mask),torch.tensor(.5))

def test_sampler_contract_k_prefix_permutation_padding():
    model=tiny().eval();n=torch.tensor([9]);x=[torch.randn(1,12,c) for c in (768,25,58)];z=torch.randn(1,32,12,24)
    outputs={k:model.sample(*x,n,k,z[:,:k],integration_steps=2) for k in (1,4,10,32)}
    for k,p in outputs.items():
        assert p.shape==(1,k,12,25) and not p[:,:,9:].any() and torch.isfinite(p).all()
        torch.testing.assert_close(p,outputs[32][:,:k],atol=2e-6,rtol=1e-5)
    p=outputs[4];order=[2,0,3,1]
    torch.testing.assert_close(model.sample(*x,n,4,z[:,order],integration_steps=2),p[:,order],atol=2e-6,rtol=1e-5)
    changed=[v.clone() for v in x]
    for v in changed:v[:,9:]=100
    zz=z[:,:4].clone();zz[:,:,9:]=100
    torch.testing.assert_close(model.sample(*changed,n,4,zz,integration_steps=2),p,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(model.sample(*x,n,4,z[:,:4],integration_steps=2,candidate_chunk=1),p,atol=2e-6,rtol=1e-5)
    assert not any(k in inspect.signature(model.sample).parameters for k in ('target','targets','session','listener_id'))
    assert (p[...,:15]>=0).all() and (p[...,:15]<=1).all() and (p[...,15:17].abs()<=1).all()
    torch.testing.assert_close(p[:,:,:9,17:].sum(-1),torch.ones(1,4,9),atol=1e-6,rtol=0)

def test_frozen_encoder_gradient_and_separate_masks():
    m=tiny();x=[torch.randn(2,11,c) for c in (768,25,58)];n=torch.tensor([11,8]);h,sm=m.condition(*x,n)
    valid=torch.arange(11)[None]<torch.tensor([4,7])[:,None];z=torch.randn(2,11,24,requires_grad=True);tau=torch.tensor([.2,.8])
    v=m.velocity(z,tau,h,valid,sm);v.square().sum().backward()
    assert not any(p.grad is not None for p in m.condition.parameters())
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0 and not v[~valid].any()
    assert m.velocity.input.weight.grad.abs().sum()>0

def test_recording_noise_identity():
    from reaction_flow.export import recording_noise
    c=FlowConfig();a=recording_noise(c,'speaker/session/source',1499,4);b=recording_noise(c,'speaker/session/source',750,1)
    torch.testing.assert_close(a[:1,:750],b,atol=0,rtol=0)
    assert not torch.equal(a[0,:749],a[0,750:1499])
    assert not torch.equal(a,recording_noise(c,'speaker/session/other',1499,4))

def test_checkpointed_rollout_preserves_gradients():
    m=tiny();x=[torch.randn(1,9,c) for c in (768,25,58)];h,mask=m.condition(*x,torch.tensor([8]));z=torch.randn(1,4,9,24)
    def evaluate(checkpointing):
        m.zero_grad(set_to_none=True);p=m.rollout(h,mask,z,16,gradient_checkpointing=checkpointing)
        weights=torch.linspace(.1,1,25);loss=(p*weights).square().sum();loss.backward()
        return p.detach(),{n:v.grad.clone() for n,v in m.velocity.named_parameters()}
    a,ga=evaluate(False);b,gb=evaluate(True);torch.testing.assert_close(a,b,atol=0,rtol=0)
    for k in ga:torch.testing.assert_close(ga[k],gb[k],atol=1e-6,rtol=1e-6)
