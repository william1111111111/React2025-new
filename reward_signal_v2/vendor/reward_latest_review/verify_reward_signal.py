"""Local synthetic checks for the reviewed reward-policy implementation.
No REACT data, checkpoint, BERT, GPU, or online request is used.
Policy math below is transcribed from reward_policy/policy.py at e7cb665.
The reward / gradient numbers in review_numbers.json are transcribed separately.
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn


def normal_log_prob(z, valid):
    return (-0.5 * (z.square() + math.log(2 * math.pi)) * valid).sum(-1)


class Coupling(nn.Module):
    def __init__(self, parity):
        super().__init__()
        self.register_buffer('fixed', torch.arange(96) % 2 == parity)
        self.net = nn.Sequential(nn.Linear(96 + 256, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 192))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def parameters_at(self, x, context, valid):
        fixed, active = self.fixed & valid, ~self.fixed & valid
        scale, shift = self.net(torch.cat((x * fixed, context), -1)).chunk(2, -1)
        return .25 * scale.tanh() * active, shift * active

    def forward(self, x, context, valid, inverse=False):
        s, t = self.parameters_at(x, context, valid)
        y = (x - t) * (-s).exp() if inverse else x * s.exp() + t
        return y * valid, (-s if inverse else s).double().sum(-1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Coupling(i % 2) for i in range(4)])

    def transform(self, z0, context, valid):
        x = z0 * valid
        ld = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.float64)
        for layer in self.layers:
            x, d = layer(x, context, valid)
            ld = ld + d
        return x, ld

    def inverse(self, z, context, valid):
        x = z * valid
        ld = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.float64)
        for layer in reversed(self.layers):
            x, d = layer(x, context, valid, True)
            ld = ld + d
        return x, ld

    def log_prob(self, z, context, valid):
        x, ld = self.inverse(z, context, valid)
        return normal_log_prob(x.double(), valid) + ld

    @torch.no_grad()
    def sample(self, z0, context, valid):
        return self.transform(z0, context, valid)[0]


def coverage(u):
    return float(np.max(u, axis=0).mean()) if len(u) else 0.


def quality_centered_advantage(q, u):
    """q is per-candidate quality utility, NOT divided by K beforehand.
    Valid for conditionally independent candidates; arrays treated as constants
    in the actor's score-function loss. Keeps the original fixed-K denominator.
    """
    q, u = np.asarray(q, np.float64), np.asarray(u, np.float64)
    K = len(q)
    if K < 2 or u.ndim != 2 or len(u) != K:
        raise ValueError('Need K>=2 and utility[K,R].')
    base = (q.sum() - q) / (K - 1)
    gains = np.array([coverage(u) - coverage(np.delete(u, k, axis=0))
                      for k in range(K)])
    return (q - base) / K + gains


def run():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    out = {}

    # One small-scale feature can collapse a 96-D Gaussian similarity.
    D2 = ((.1 / .001) ** 2) / 96
    sim = float(np.exp(-.5 * D2))
    assert 0 < sim < 1e-20
    out['kernel_single_coordinate_example'] = dict(D2=D2, similarity=sim,
        description='Synthetic: a .1 difference / .001 scale in one of 96 dimensions.')

    # Positive but tiny is not an informative reward; large quality sums hide it.
    u = np.diag([1e-70, 2e-70])
    q = np.array([-3., -4.])
    pure = np.array([coverage(u) - coverage(np.delete(u, k, 0)) for k in range(2)])
    score = lambda ix: coverage(u[ix]) + q[ix].sum() / 2
    mixed = np.array([score(np.arange(2)) - score(np.array([1-k])) for k in range(2)])
    assert (u.max(1) > 0).all() and (pure > 0).all()
    assert np.array_equal(mixed, q / 2)
    out['tiny_positive_not_signal'] = dict(u_positive_pass=True,
        coverage_gains=pure.tolist(), mixed_gains=mixed.tolist(),
        quality_only_gains=(q / 2).tolist(),
        lost_after_adding_quality=True)

    # All gated entries zero: wider bandwidth alone cannot fix a closed gate.
    gate = np.zeros((3, 4), dtype=bool)
    assert coverage(gate * np.ones((3, 4)) * .5) == 0.
    out['all_closed_gates'] = 'bandwidth repair cannot create eligible pairs'

    # Calibrate a global Gaussian denominator to a target median similarity.
    distances = np.array([100., 200., 300.])
    h2 = float(np.median(distances) / (2 * np.log(2.)))
    new_sim = np.exp(-distances / (2 * h2))
    assert np.isclose(np.median(new_sim), .5)
    out['bandwidth_math_only'] = dict(h2=h2, similarities=new_sim.tolist())

    # Exact enumeration: additive quality baseline leaves expected gradient
    # unchanged, while removing a large action-independent offset reduces noise.
    p, K = .3, 3
    records = []
    for bits in itertools.product([0., 1.], repeat=K):
        a = np.array(bits)
        probability = float(np.prod(p ** a * (1-p) ** (1-a)))
        q = -100. + a
        grad_score = a - p  # score wrt common Bernoulli logit
        old = float(np.dot(q / K, grad_score))
        baseline = (q.sum() - q) / (K - 1)
        new = float(np.dot((q - baseline) / K, grad_score))
        records.append((probability, old, new))
    mean_old = sum(w*x for w,x,y in records)
    mean_new = sum(w*y for w,x,y in records)
    var_old = sum(w*(x-mean_old)**2 for w,x,y in records)
    var_new = sum(w*(y-mean_new)**2 for w,x,y in records)
    assert np.isclose(mean_old, p*(1-p)) and np.isclose(mean_new, mean_old)
    assert var_new < var_old
    out['quality_baseline_exact_enumeration'] = dict(expected_old=mean_old,
        expected_centered=mean_new, variance_old=var_old, variance_centered=var_new,
        caveat='Illustrative toy distribution; not a measured variance reduction on REACT.')

    # Nonlinear set coverage: own-action-independent LOO baseline remains valid.
    direct, loo = 0., 0.
    for bits in itertools.product([0, 1], repeat=K):
        a = np.array(bits)
        probability = float(np.prod(p ** a * (1-p) ** (1-a)))
        U = np.eye(2)[a]
        R = coverage(U)
        gains = np.array([R - coverage(np.delete(U, k, 0)) for k in range(K)])
        direct += probability * R * (a-p).sum()
        loo += probability * np.dot(gains, a-p)
    assert np.isclose(direct, loo)
    out['coverage_LOO_unbiased_toy'] = dict(direct=direct, leave_one_out=loo)

    # Same no-grad sampling and detached action: score-function still updates.
    policy = Policy().double()
    z0 = torch.randn(20, 96, dtype=torch.float64)
    ctx = torch.randn(20, 256, dtype=torch.float64)
    valid = torch.ones_like(z0, dtype=torch.bool)
    action = policy.sample(z0, ctx, valid)
    assert not action.requires_grad and torch.equal(action, z0)
    before = torch.cat([x.detach().flatten() for x in policy.parameters()]).clone()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=0.)
    reward = action[:, 0].detach()
    loss = -(reward * policy.log_prob(action.detach(), ctx, valid)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.))
    assert gn > 0
    opt.step()
    after = torch.cat([x.detach().flatten() for x in policy.parameters()])
    delta = float((after-before).norm())
    assert delta > 0
    out['detached_action_policy_update_CPU'] = dict(gradient_norm=gn,
        parameter_delta_l2=delta, action_requires_grad=action.requires_grad,
        caveat='Transcribed policy, synthetic reward, no REACT generator or checkpoint.')

    # Orthonormal low-frequency replacement is not an output-diversity bound.
    out['orthonormal_action_geometry'] = {}
    for N in (750, 3000):
        out['orthonormal_action_geometry'][str(N)] = dict(
            one_DC_coefficient_shift_point1_per_frame=.1/math.sqrt(N),
            four_basis_fraction=4/N)
    out['passed_checks'] = 7
    return out

if __name__ == '__main__':
    result = run()
    Path(__file__).with_name('synthetic_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
