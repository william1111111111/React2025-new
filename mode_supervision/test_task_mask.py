import torch
from .task_mask import masked_task_parts
from reaction_flow.task_dynamics_train import task_parts


def test_mixed_short_and_valid():
    torch.manual_seed(1)
    pred=torch.rand(2,2,3,25,requires_grad=True)
    b=dict(source_lengths=torch.tensor([1,3]),pair_lengths=torch.tensor([1,3]),_target_lengths=torch.tensor([[1,1],[3,3]]),_targets=torch.rand(2,2,3,25),paired_target=torch.rand(2,3,25))
    loss,parts=masked_task_parts(pred,b)
    expected,_=task_parts(pred[1:],{k:v[1:] for k,v in b.items()})
    assert torch.equal(loss,expected/2)
    loss.backward();assert pred.grad[0].abs().sum()==0 and torch.isfinite(pred.grad).all()
    assert parts['task_unsupported_samples']==1
    zero,_=masked_task_parts(pred[:1],{k:v[:1] for k,v in b.items()});assert zero==0 and zero.requires_grad
