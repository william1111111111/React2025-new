import torch

from coreset_reactor.eval_overlap_save_student import (
    overlap_save,
    powered_center_overlap,
)
from coreset_reactor.eval_student_window_gap import (
    WINDOW,
    center_weights,
    overlap_add,
)


def test_single_window_is_exact_identity():
    prediction = torch.rand(1, 10, WINDOW, 25)
    result, selected, switches = overlap_save(prediction, [0], WINDOW)
    assert torch.equal(result, prediction[0])
    assert torch.equal(selected, torch.zeros(WINDOW, dtype=torch.long))
    assert switches == []


def test_each_frame_uses_largest_center_weight_without_averaging():
    length = WINDOW + 375
    predictions = torch.stack((
        torch.zeros(10, WINDOW, 25),
        torch.ones(10, WINDOW, 25),
    ))
    result, selected, switches = overlap_save(predictions, [0, 375], length)
    weights = center_weights(WINDOW)
    expected = torch.zeros(length, dtype=torch.long)
    for frame in range(length):
        candidates = []
        for index, start in enumerate((0, 375)):
            if start <= frame < start + WINDOW:
                candidates.append((float(weights[frame - start]), -index, index))
        expected[frame] = max(candidates)[2]
    assert torch.equal(selected, expected)
    assert torch.equal(result[:, :, 0], expected[None].expand(10, -1))
    assert switches == [int((expected[1:] != expected[:-1]).nonzero()[0]) + 1]


def test_power_one_matches_original_center_weighting():
    torch.manual_seed(4)
    predictions = torch.rand(2, 10, WINDOW, 25)
    starts = [0, 375]
    expected = overlap_add(predictions, starts, WINDOW + 375, "center")
    actual = powered_center_overlap(predictions, starts, WINDOW + 375, 1)
    torch.testing.assert_close(actual, expected)
