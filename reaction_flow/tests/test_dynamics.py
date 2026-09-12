import torch
from reaction_flow.dynamics import p8,spread,decomposition,dynamic_loss
from mam_target.losses import grid

def test_grid_support():
    x=torch.randn(1,750,25,requires_grad=True);grid(x).sum().backward()
    assert int(x.grad.abs().sum((0,2)).gt(0).sum())==62

def test_orthogonal_tail():
    y=torch.randn(10,19,25,dtype=torch.double);p=p8(y)
    assert p.shape==y.shape
    torch.testing.assert_close(p[:,16:],y[:,16:].mean(1,keepdim=True).expand(-1,3,-1))
    full,parts=decomposition(y);torch.testing.assert_close(full,sum(parts.values()),atol=1e-14,rtol=0)

def test_dynamics_masks_oracle_and_blind_frames():
    target=torch.randn(2,35,25);pred=target.clone().requires_grad_();sc=torch.ones(3,3)
    assert dynamic_loss(pred,target,[35,2],torch.tensor([.7,.1]),sc)==0
    changed=target.clone();changed[0,7]+=1
    assert dynamic_loss(changed,target,[35,2],torch.tensor([.7,.1]),sc)>0
    changed=target.clone();changed[0,20:]+=100
    assert dynamic_loss(changed,target,[20,2],torch.tensor([.7,.1]),sc)==0
    zero=dynamic_loss(pred,target,[1,1],torch.tensor([.7,.1]),sc);zero.backward();assert pred.grad.eq(0).all()
    u=torch.randn(2,35,24);e=torch.randn_like(u);t=.7
    torch.testing.assert_close((1-t)*e+t*u+(1-t)*(u-e),u)

def test_task_structure_matches_existing():
    from reaction_flow.task_dynamics_train import task_parts
    from reaction_flow.finetune import task_loss
    torch.manual_seed(7)
    p=torch.rand(1,4,20,25,requires_grad=True);t=torch.rand(1,4,20,25)
    b={'source_lengths':torch.tensor([20]),'pair_lengths':torch.tensor([17]),'paired_target':t[:,0], '_targets':t,'_target_lengths':torch.tensor([[17,20,19,18]])}
    new,_=task_parts(p,b);old=task_loss(p,b,t,b['_target_lengths'])
    torch.testing.assert_close(new,old,atol=0,rtol=0)
    g=torch.autograd.grad(new,p)[0];assert torch.isfinite(g).all() and g.abs().sum()>0
