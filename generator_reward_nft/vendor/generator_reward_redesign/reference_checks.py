"""CPU-only algebra/gradient checks for a proposed generator-level reward update.
No REACT checkpoint, private data, BERT weights, or actual task metrics are loaded.
Core positive/negative objective follows DiffusionNFT (arXiv:2509.16117),
expressed with the REACT noise-to-data time convention.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import Tensor


def masked_mse(x: Tensor, y: Tensor, valid: Tensor) -> Tensor:
    if x.shape != y.shape or x.ndim != 3 or valid.shape != x.shape[:2]:
        raise ValueError('Expected [B,T,D] tensors and [B,T] mask')
    counts = valid.sum(1) * x.shape[-1]
    if (counts == 0).any():
        raise ValueError('Empty sample')
    a = x.masked_fill(~valid[..., None], 0)
    b = y.masked_fill(~valid[..., None], 0)
    return (a - b).square().sum((1, 2)) / counts


def nft_core(student_v: Tensor, old_v: Tensor, target_v: Tensor,
             reward_weight: Tensor, valid: Tensor, beta: float = 1.) -> Tensor:
    """Core velocity-MSE form, not a complete production trainer.
    'reward_weight' is a bounded training weight, NOT calibrated correctness.
    All targets/teacher/rewards are detached. No sample log-density is used.
    """
    if beta <= 0 or reward_weight.shape != (student_v.shape[0],):
        raise ValueError('Invalid beta or weights')
    if not torch.isfinite(reward_weight).all() or not ((reward_weight >= 0) & (reward_weight <= 1)).all():
        raise ValueError('Weights must be finite in [0,1]')
    old, target, r = old_v.detach(), target_v.detach(), reward_weight.detach()
    positive = (1 - beta) * old + beta * student_v
    negative = (1 + beta) * old - beta * student_v
    return (r * masked_mse(positive, target, valid) +
            (1 - r) * masked_mse(negative, target, valid)).mean()


def run_checks() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(123)
    checks = {}
    mask = torch.ones((1, 1), dtype=torch.bool)
    for name, r, expected in [('positive', 1., -2.), ('neutral', .5, 0.), ('negative', 0., 2.)]:
        v = torch.tensor([[[0.]]], dtype=torch.float64, requires_grad=True)
        old = torch.zeros_like(v, requires_grad=True)
        target = torch.ones_like(v, requires_grad=True)
        weight = torch.tensor([r], dtype=torch.float64, requires_grad=True)
        loss = nft_core(v, old, target, weight, mask)
        loss.backward()
        assert torch.allclose(v.grad, torch.full_like(v, expected))
        assert old.grad is None and target.grad is None and weight.grad is None
        checks[name + '_gradient'] = float(v.grad.item())
    checks['teacher_target_reward_detached'] = True

    v = torch.randn(2, 3, 4, dtype=torch.float64)
    u = torch.randn_like(v)
    eps = torch.randn_like(v)
    tau = torch.tensor([.2, .8], dtype=v.dtype)[:, None, None]
    state = (1 - tau) * eps + tau * u
    reconstructed = state + (1 - tau) * (u - eps)
    assert torch.allclose(reconstructed, u, atol=1e-12, rtol=0)
    checks['noise_to_data_endpoint_sign'] = True

    val = torch.tensor([[True, True, False], [True, False, False]])
    raw = masked_mse(v, u, val)
    v2 = v.clone(); u2 = u.clone()
    v2[~val] = float('nan'); u2[~val] = 1.e20
    assert torch.equal(raw, masked_mse(v2, u2, val))
    checks['padding_does_not_enter_loss'] = True

    # A minimal trainable vector field: black-box rewards are detached,
    # yet the vector-field network gets updated and fixed-noise outputs change.
    student = torch.nn.Linear(2, 1, bias=True).double()
    old = torch.nn.Linear(2, 1, bias=True).double(); old.load_state_dict(student.state_dict())
    old.requires_grad_(False)
    optim = torch.optim.SGD(student.parameters(), lr=.05)
    noisy_state = torch.tensor([[.1, .2], [.3, .4]], dtype=torch.float64)
    before = student(noisy_state).detach().clone()
    desired = before + torch.tensor([[1.], [-1.]], dtype=torch.float64)
    pred = student(noisy_state)[:, None]
    prior = old(noisy_state).detach()[:, None]
    loss = nft_core(pred, prior, desired[:, None], torch.tensor([1., 0.], dtype=torch.float64),
                    torch.ones(2, 1, dtype=torch.bool))
    optim.zero_grad(); loss.backward()
    gnorm = sum(p.grad.square().sum() for p in student.parameters()).sqrt()
    assert gnorm > 0
    optim.step()
    after = student(noisy_state).detach()
    assert not torch.equal(before, after)
    assert all(p.grad is None for p in old.parameters())
    checks['toy_vector_field_grad_norm'] = float(gnorm)
    checks['toy_fixed_input_output_change_rms'] = float((after - before).square().mean().sqrt())

    # Exact norm identity for modifying an orthonormal low-frequency subspace.
    n, r, k, d = 3000, 4, 10, 24
    q, _ = torch.linalg.qr(torch.randn(n, r, dtype=torch.float64))
    delta = torch.randn(k, r, d, dtype=torch.float64)
    perturbation = torch.einsum('nr,krd->knd', q, delta)
    ratio = perturbation.square().mean().sqrt() / delta.square().mean().sqrt()
    theoretical = (r / n) ** .5
    assert abs(float(ratio) - theoretical) < 1.e-12
    checks['orthonormal_input_RMS_ratio_n3000_r4'] = float(ratio)
    checks['theoretical_RMS_ratio'] = theoretical
    return {'scope': 'CPU synthetic algebra and autograd only; no REACT experiment',
            'checks_passed': 8, 'checks': checks}

if __name__ == '__main__':
    out = run_checks()
    Path(__file__).with_name('reference_checks_results.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
