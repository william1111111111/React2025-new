"""CPU mathematical checks only. No REACT media, checkpoints, or RL training."""
import itertools
import json
from pathlib import Path
import numpy as np


def coverage(q: np.ndarray) -> float:
    """q[K,J] is already quality-gated, nonnegative reference support."""
    assert q.ndim == 2 and q.shape[1] > 0
    return float(q.max(axis=0).mean()) if len(q) else 0.0


def run() -> dict:
    checks = {}
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(31, 4))
    b, r = np.linalg.qr(raw)
    b *= np.where(np.diag(r) >= 0, 1.0, -1.0)[None]
    eps = rng.normal(size=(31, 3))
    z0 = b.T @ eps
    initial = eps + b @ (z0 - z0)
    checks['identity_policy_preserves_initial_noise_exactly'] = bool(np.array_equal(eps, initial))
    z = z0 + rng.normal(size=z0.shape) * 0.2
    changed = eps + b @ (z - z0)
    projection_error = float(np.max(np.abs(b.T @ changed - z)))
    residual_error = float(np.max(np.abs((changed-b@(b.T@changed))-(eps-b@z0))))
    checks['new_action_is_correct_projected_coefficient'] = projection_error < 1e-12
    checks['uncontrolled_noise_residual_is_unchanged'] = residual_error < 1e-12

    a = np.array([[1., 0., 0.]])
    duplicate = np.vstack((a, a, a))
    added = np.vstack((duplicate, [0., 1., 0.]))
    junk = np.vstack((added, [0., 0., 0.]))
    checks['duplicate_has_no_extra_reference_coverage'] = coverage(duplicate) == coverage(a)
    checks['new_qualified_mode_increases_coverage'] = coverage(added) > coverage(duplicate)
    checks['quality_rejected_outlier_gets_no_coverage_credit'] = coverage(junk) == coverage(added)

    # Exact enumeration of an iid categorical group policy. The group return has
    # both quality penalties and a non-additive coverage bonus. A counterfactual
    # baseline excluding action k is independent of that action conditional on x.
    p = np.array([.70, .16, .10, .04])
    k = 4
    features = np.vstack((np.eye(3), np.zeros((1,3))))
    def ret(seq):
        # Keep the quality denominator fixed at K for counterfactual subsets.
        return coverage(features[list(seq)]) - 2.0 * sum(v == 3 for v in seq) / k
    direct = np.zeros(4); difference = np.zeros(4)
    for seq in itertools.product(range(4), repeat=k):
        mass = float(np.prod(p[list(seq)])); total = ret(seq)
        for i, action in enumerate(seq):
            score = np.eye(4)[action] - p
            without = seq[:i] + seq[i+1:]
            direct += mass * total * score
            difference += mass * (total-ret(without)) * score
    gradient_error = float(np.max(np.abs(direct-difference)))
    checks['leave_one_out_matches_exact_group_policy_gradient'] = gradient_error < 1e-12
    common_rewards = np.full(k, .6)
    checks['identical_group_reward_centered_within_group_is_zero'] = bool(
        np.all(common_rewards-common_rewards.mean() == 0))

    # Equal correct rewards retain skewed reference ratios in a KL-regularized
    # optimum. This does not promote equal coverage just because both are correct.
    rewards = np.array([.9,.9,.9,.05]); beta=.1
    logits = np.log(p)+rewards/beta
    optimum=np.exp(logits-logits.max());optimum/=optimum.sum()
    checks['equal_correct_rewards_do_not_remove_reference_skew'] = bool(
        np.isclose(optimum[0]/optimum[1], p[0]/p[1]))
    assert all(checks.values()), checks
    return dict(scope='synthetic mathematical checks only; no model experiments',
                tests=len(checks), passed=sum(checks.values()), checks=checks,
                max_projected_coefficient_error=projection_error,
                max_residual_error=residual_error,
                exact_gradient_error=gradient_error,
                categorical_group_gradient=direct.tolist(),
                kl_regularized_equal_quality_probabilities=optimum.tolist())

if __name__ == '__main__':
    out=run(); path=Path(__file__).with_name('verification_results.json')
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
