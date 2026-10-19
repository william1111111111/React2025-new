import numpy as np
import torch

from coreset_reactor.common_va import (feasible_shift, apply_shift, ccc_channels,
    SharedVAQualityPool, strict_choice)
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import _rolling_frd


def sample():
    rng = np.random.default_rng(41)
    pred = torch.from_numpy(rng.random((10, 12, 25)).astype('float32'))
    target = torch.from_numpy(rng.random((3, 12, 25)).astype('float32'))
    pred[..., :15] = pred[..., :15].round()
    return pred, target


def test_shared_projection_preserves_geometry_and_channels():
    pred, _ = sample()
    desired = torch.linspace(-3, 3, 24).reshape(12, 2)
    shift = feasible_shift(pred, desired)
    out = apply_shift(pred, shift)
    assert out[..., 15:17].abs().max() <= 1+1e-7
    assert torch.equal(pred[..., :15], out[..., :15])
    assert torch.equal(pred[..., 17:], out[..., 17:])
    torch.testing.assert_close(out-out.mean(0), pred-pred.mean(0), atol=3e-7, rtol=0)


def test_ccc_objective_matches_native_frc_with_constants():
    pred, target = sample()
    pred[..., 0], target[..., 1] = 0, 1
    actual = ccc_channels(target.double(), pred.double()).mean(-1).max(-1).values.numpy()
    np.testing.assert_allclose(actual, candidate_frc(target.numpy(), pred.numpy()), atol=1e-7, rtol=1e-5)


def test_invariant_group_cache_matches_original_rolling_dtw():
    pred, target = sample()
    shifted = apply_shift(pred, feasible_shift(pred, torch.full((12, 2), -.2)))
    pool = SharedVAQualityPool(workers=2, max_targets=3)
    try:
        actual, calls = pool.score_variants([pred, shifted], target)
        expected = [[_rolling_frd(target.numpy(), x.numpy()) for x in output] for output in [pred, shifted]]
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
        assert calls[0] == 30 and calls[2] <= 30
    finally:
        pool.close()


def test_strict_oracle_falls_back_without_relaxing_quality():
    assert strict_choice([1, 1.1, .9], [10, 11, 9]) == 0
    assert strict_choice([1, 1.1, 1.2], [10, 10, 9]) == 2
