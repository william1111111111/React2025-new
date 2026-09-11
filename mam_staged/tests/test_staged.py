import torch
from mam_staged.losses import *
from mam_staged.params import *
from mam_target.model import make_model
from hirp.config import HiRPConfig

def inputs():
    torch.manual_seed(3);p=torch.randn(2,4,19,25,dtype=torch.double,requires_grad=True);parent=p.detach()*.5;y=torch.randn(2,4,19,25,dtype=torch.double);sl=torch.tensor([19,17]);tl=torch.tensor([[18,19,19,19],[17,17,17,17]]);ids=[['a','b','b','c'],['d','d','d','d']]
    return p,parent,y,sl,tl,ids

def test_disp_permutations_padding_unique():
    p,q,y,s,t,ids=inputs();a=dispersions(p,q,y,s,t,ids);assert not a[3][1].any()
    order=[2,0,3,1];z=dispersions(p[:,order],q[:,order],y[:,:,],s,t,ids)
    for x,v in zip(a,z):torch.testing.assert_close(x,v)
    ids2=[[row[j] for j in order] for row in ids];y[:,2]=y[:,1] # duplicate ID must represent same content
    a=dispersions(p,q,y,s,t,ids);b=dispersions(p,q,y[:,order],s,t[:,order],ids2)
    for x,v in zip(a,b):torch.testing.assert_close(x,v)
    pp=p.detach().clone();pp[0,:,18:]=100;pp[1,:,17:]=100
    torch.testing.assert_close(a[0],dispersions(pp,q,y,s,t,ids)[0])

def test_dynamic_and_self_exclusion():
    x=torch.randn(4,17,3,dtype=torch.double);pooled=pool8(x,17);assert pooled.shape[1]==3
    expected=torch.stack([(pooled[i]-pooled[j]).square().mean() for i in range(4) for j in range(4) if i!=j]).mean();torch.testing.assert_close(spread(pooled),expected)
    offsets=torch.randn(4,1,3,dtype=torch.double);a=pooled-pooled.mean(1,keepdim=True);b=pool8(x+offsets,17);b=b-b.mean(1,keepdim=True);torch.testing.assert_close(spread(a),spread(b))
    assert spread(a)>0

def test_guard_score_only():
    a=torch.tensor([.2,.3],requires_grad=True);ref=a.detach().clone().requires_grad_();cal={'s_C':.1,'s_D':.2}
    guard,*_=quality_guard(a,a,ref,ref,cal);assert guard==0
    loss,*_=quality_guard(a+.1,a,ref,ref,cal);loss.backward();assert ref.grad is None and torch.isfinite(a.grad).all()

def test_frozen_groups_and_stages():
    m=make_model(config=HiRPConfig(d_model=16,nhead=4,num_layers=1,dim_feedforward=32,dropout=.1,dilations=(1,2)))
    opt=make_optimizer(m);apply_stage(m,opt,'S2_staged',0);before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    x=[torch.randn(1,12,c) for c in (768,25,58)];n=torch.tensor([10]);z=torch.randn(1,4,32)
    p=m(*x,n,4,z);p.square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in partition(m)['stochastic'])
    opt.step()
    for name,v in m.named_parameters():
        if name in before:torch.testing.assert_close(v,before[name],atol=0,rtol=0)
    assert not m.training
    apply_stage(m,opt,'S2_staged',500);assert all(p.requires_grad for _,p in partition(m)['decoder_body']) and not any(p.requires_grad for _,p in partition(m)['source_base'])
    apply_stage(m,opt,'S2_staged',1500);assert opt.param_groups[2]['lr']==2.5e-5

def test_original_r2_objective_and_all_candidates():
    from mam_target.losses import objective,pair_costs
    torch.manual_seed(44)
    pred=torch.randn(1,4,12,25,requires_grad=True)
    target=torch.randn(1,4,12,25)
    batch={'paired_target':target[:,0], 'pair_lengths':torch.tensor([9]), 'source_lengths':torch.tensor([12])}
    lengths=torch.tensor([[9,12,11,10]])
    weights=dict(a=1.,b=.15813484622234084,beta=.25,gamma=.25,eta=.5,grid=4)
    c,d=pair_costs(pred,target,batch['source_lengths'],lengths,4)
    new=r2_terms(pred,batch,weights,c,d);old=objective(pred,batch,target,lengths,weights)
    torch.testing.assert_close(new['r2'],old['loss'],atol=0,rtol=0)
    grad=torch.autograd.grad(new['r2'],pred)[0]
    assert torch.isfinite(grad).all() and (grad.abs().sum((0,2,3))>0).all()
