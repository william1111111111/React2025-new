import pytest
import torch

from coreset_reactor.train_hybrid_teacher import HybridTargetLoss
from regnn.conditional_model import ConditionalREGNNLoss


def example():
    generator = torch.Generator().manual_seed(712)
    prediction = torch.rand(2, 8, 25, generator=generator, requires_grad=True)
    raw = torch.rand(2, 8, 25, generator=generator)
    processed = torch.rand(2, 8, 25, generator=generator)
    mask = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
    velocity = torch.zeros(2, dtype=torch.bool)
    return prediction, raw, processed, mask, velocity


def test_zero_weight_is_exact_baseline_loss_and_gradient():
    pred, raw, proc, mask, velocity = example()
    baseline = ConditionalREGNNLoss()(pred, raw, mask, velocity)
    hybrid = HybridTargetLoss(auxiliary_weight=0)(pred, torch.cat((raw, proc), -1), mask, velocity)
    for key in baseline:
        assert torch.equal(baseline[key], hybrid[key])
    first = torch.autograd.grad(baseline['loss'], pred, retain_graph=True)[0]
    second = torch.autograd.grad(hybrid['loss'], pred)[0]
    assert torch.equal(first, second)


def test_auxiliary_is_added_without_reweighting_raw_objective():
    pred, raw, proc, mask, velocity = example()
    baseline = ConditionalREGNNLoss()(pred, raw, mask, velocity)
    auxiliary = ConditionalREGNNLoss(velocity_weight=0)(pred, proc, mask, velocity)
    expected = baseline['loss'] + 0.1*auxiliary['loss']
    actual = HybridTargetLoss()(pred, torch.cat((raw, proc), -1), mask, velocity)
    assert torch.equal(actual['loss'], expected)
    assert torch.equal(torch.autograd.grad(expected, pred, retain_graph=True)[0],
                       torch.autograd.grad(actual['loss'], pred)[0])
    for key in ('mse', 'ccc', 'ccc_loss', 'velocity'):
        assert torch.equal(actual[key], baseline[key])


def test_padding_does_not_change_hybrid_loss():
    pred, raw, proc, mask, velocity = example()
    packed = torch.cat((raw, proc), -1)
    original = HybridTargetLoss()(pred, packed, mask, velocity)['loss']
    packed[~mask] = 1000
    assert torch.equal(original, HybridTargetLoss()(pred, packed, mask, velocity)['loss'])


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(ValueError):
        HybridTargetLoss(auxiliary_weight=weight)
