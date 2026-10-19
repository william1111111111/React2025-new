import numpy as np
import pytest

from coreset_reactor.eval_random_reference_frc import draw_indices
from coreset_reactor.fast_frc import candidate_frc, pairwise_frc


@pytest.mark.parametrize('count', [1, 5, 10, 100])
@pytest.mark.parametrize('policy', ['paired_plus9', 'uniform10'])
def test_sampling_contract(count, policy):
    indices = draw_indices(count, 'speaker/session0/example', 1234, policy)
    assert len(indices) == 10 and all(0 <= x < count for x in indices)
    assert indices == draw_indices(count, 'speaker/session0/example', 1234, policy)
    if count >= 10:
        assert len(set(indices)) == 10
    if policy == 'paired_plus9':
        assert indices[0] == 0
        if count > 1:
            assert 0 not in indices[1:]


def test_pair_matrix_matches_single_reference_and_best_reference():
    rng = np.random.default_rng(3)
    targets = rng.normal(size=(11, 50, 25)).astype(np.float32)
    pred = rng.normal(size=(10, 50, 25)).astype(np.float32)
    targets[0, :, :4] = 0
    pred[1, :, 3:7] = 1
    matrix = pairwise_frc(targets, pred)
    assert matrix.shape == (10, 11)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix.max(-1), candidate_frc(targets, pred), atol=1e-12)
    for j in range(len(targets)):
        np.testing.assert_allclose(matrix[:, j], candidate_frc(targets[j:j+1], pred), atol=1e-12)
    chosen = np.array([draw_indices(11, 'x', seed, 'paired_plus9') for seed in range(4)])
    fast = matrix[:, chosen].max(-1).sum(0)
    for i in range(4):
        np.testing.assert_allclose(fast[i], candidate_frc(targets[chosen[i]], pred).sum(), atol=1e-12)
