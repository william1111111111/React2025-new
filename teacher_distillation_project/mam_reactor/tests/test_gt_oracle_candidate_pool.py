import numpy as np
import pytest
import torch

from framework.metrics import compute_FRC, compute_FRD, compute_s_mse
from regnn.probe_gt_oracle_candidate_pool import (
    pairwise_official_frdiv,
    quality_constrained_local_search,
    score_candidate_pool,
    set_diversity,
)


def test_strict_quality_swap_improves_diversity():
    pairwise = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    frc = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    frd = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    slots = [0, 1, 1]

    # Pad the toy problem to the fixed ten-slot selector contract.
    big_pairwise = np.zeros((12, 12), dtype=np.float64)
    big_pairwise[:3, :3] = pairwise
    baseline = [0] + list(range(2, 11))
    # slot-1 baseline is index 1; alternative is index 11
    baseline[1] = 1
    big_pairwise[0, 11] = big_pairwise[11, 0] = 2.0
    for idx in baseline[2:]:
        big_pairwise[1, idx] = big_pairwise[idx, 1] = 1.0
        big_pairwise[11, idx] = big_pairwise[idx, 11] = 3.0
    frc_big = np.ones(12, dtype=np.float64)
    frd_big = np.ones(12, dtype=np.float64)
    slots_big = [0, 1] + list(range(2, 10)) + [9, 1]

    result = quality_constrained_local_search(
        big_pairwise,
        frc_big,
        frd_big,
        baseline,
        slots_big,
        enable_two_swap=True,
    )
    assert result["selected_indices"][0] == 0
    assert 11 in result["selected_indices"]
    assert result["selected_frc"] >= result["baseline_frc"]
    assert result["selected_frd"] <= result["baseline_frd"]
    assert result["selected_frdiv"] > result["baseline_frdiv"]


def test_quality_constraint_rejects_lower_frc():
    n = 11
    pairwise = np.zeros((n, n), dtype=np.float64)
    baseline = list(range(10))
    # Candidate 10 is a very diverse replacement for slot 1.
    pairwise[10, :10] = 10.0
    pairwise[:10, 10] = 10.0
    frc = np.ones(n, dtype=np.float64)
    frc[10] = 0.9
    frd = np.ones(n, dtype=np.float64)
    slots = list(range(10)) + [1]

    result = quality_constrained_local_search(
        pairwise,
        frc,
        frd,
        baseline,
        slots,
        frc_slack=0.0,
        frd_slack=0.0,
    )
    assert result["selected_indices"] == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FRDiv matrix helper uses CUDA")
def test_frdiv_matrix_matches_official_smse():
    generator = torch.Generator().manual_seed(7)
    prediction = torch.rand((10, 8, 25), generator=generator)
    prediction[:, :, :15] = torch.round(prediction[:, :, :15])
    candidates = [prediction[index] for index in range(10)]
    pairwise = pairwise_official_frdiv(candidates, torch.device("cuda"))
    from_matrix = set_diversity(pairwise, list(range(10)))
    official = float(compute_s_mse([prediction]))
    assert from_matrix == pytest.approx(official, abs=1e-5, rel=1e-5)


def test_candidate_quality_labels_match_official_metrics():
    generator = torch.Generator().manual_seed(11)
    prediction = torch.rand((10, 6, 25), generator=generator)
    target = torch.rand((10, 6, 25), generator=generator)
    prediction[:, :, :15] = torch.round(prediction[:, :, :15])
    target[:, :, :15] = torch.round(target[:, :, :15])

    candidates = [prediction[index] for index in range(10)]
    frc, frd = score_candidate_pool(candidates, target, workers=1)

    official_frc = float(compute_FRC([prediction], [target], p=1))
    official_frd = float(compute_FRD([prediction], [target], p=1))

    assert float(frc.sum()) == pytest.approx(official_frc, abs=1e-6, rel=1e-6)
    assert float(frd.sum()) == pytest.approx(official_frd, abs=1e-5, rel=1e-5)
