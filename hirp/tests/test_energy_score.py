import pytest
import torch
from hirp.losses import paired_energy_score


def test_hand_calculation():
    p = torch.zeros(1, 2, 3, 25)
    p[:, 1] = 2
    target = torch.ones(1, 3, 25)
    mask = torch.tensor([[True, True, False]])
    p[:, :, 2] = float('nan')
    target[:, 2] = float('nan')
    scale = torch.full((25,), 2.)
    result = paired_energy_score(p, target, mask, scale)
    expected = torch.tensor(0.25 + 1e-8).sqrt() - 0.5 * torch.tensor(1 + 1e-8).sqrt()
    torch.testing.assert_close(result, expected)


def test_reference_and_mask_gradient():
    p = torch.randn(2, 4, 5, 25, requires_grad=True)
    y = torch.randn(2, 5, 25)
    mask = torch.arange(5)[None] < torch.tensor([5, 3])[:, None]
    scale = torch.linspace(.5, 2, 25)
    terms = []
    for b in range(2):
        q, target = p[b, :, mask[b]] / scale, y[b, mask[b]] / scale
        dist = lambda a, b: ((a-b).square().mean() + 1e-8).sqrt()
        cross = torch.stack([dist(v, target) for v in q]).mean()
        self_term = torch.stack([dist(q[i], q[j]) for i in range(4) for j in range(4) if i != j]).mean()
        terms.append(cross - .5*self_term)
    actual = paired_energy_score(p, y, mask, scale)
    torch.testing.assert_close(actual, torch.stack(terms).mean())
    actual.backward()
    assert torch.count_nonzero(p.grad[1, :, 3:]) == 0


def test_k1_rejected():
    with pytest.raises(ValueError, match='K >= 2'):
        paired_energy_score(torch.zeros(1, 1, 2, 25), torch.zeros(1, 2, 25), torch.ones(1, 2, dtype=torch.bool))


@pytest.mark.parametrize('scale', [torch.zeros(25), -torch.ones(25), torch.ones(24), torch.full((25,), float('nan'))])
def test_bad_scale(scale):
    with pytest.raises(ValueError):
        paired_energy_score(torch.zeros(1, 2, 2, 25), torch.zeros(1, 2, 25), torch.ones(1, 2, dtype=torch.bool), scale)


def test_amp_float32():
    p = torch.randn(1, 3, 4, 25).bfloat16().requires_grad_()
    y = torch.randn(1, 4, 25).bfloat16()
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = paired_energy_score(p, y, mask)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_sample_permutation_and_translation_invariance():
    predictions = torch.randn(2, 4, 5, 25)
    target = torch.randn(2, 5, 25)
    mask = torch.tensor([[True]*5, [True]*3 + [False]*2])
    original = paired_energy_score(predictions, target, mask)
    permuted = paired_energy_score(predictions[:, [2, 0, 3, 1]], target, mask)
    shift = torch.randn(2, 5, 25)
    translated = paired_energy_score(predictions + shift[:, None], target + shift, mask)
    torch.testing.assert_close(original, permuted)
    torch.testing.assert_close(original, translated)
