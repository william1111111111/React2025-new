import numpy as np

from coreset_reactor.conditional_slot_manifest import (
    assign_set, crossfit_predictions, shuffled_condition_indices,
)


def test_common_condition_shift_cancels_for_fixed_set():
    rng = np.random.default_rng(8)
    values, centers = rng.normal(size=(2, 9, 15))
    shift = rng.normal(size=(1, 15))*100
    a = assign_set(values, centers, centers, anchor_weight=0)
    b = assign_set(values, centers+shift, centers, anchor_weight=0)
    assert np.array_equal(a, b)


def test_mode_specific_prediction_can_change_assignment_without_changing_set():
    values = np.eye(9)
    centers = values.copy()
    order = np.roll(np.arange(9), 1)
    result = assign_set(values, values[order], centers)
    assert np.array_equal(result, order)
    assert sorted(result) == list(range(9))


def test_crossfit_never_fits_heldout_session_targets():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 6))
    y = rng.normal(size=(40, 3, 5))
    sessions = np.repeat(np.arange(8), 5)
    pred, _, splits, _ = crossfit_predictions(x, y, sessions, components=4)
    train, test = splits[0]
    changed = y.copy()
    changed[test] += 1000
    pred2, _, _, _ = crossfit_predictions(x, changed, sessions, components=4)
    np.testing.assert_array_equal(pred[test], pred2[test])
    assert not set(sessions[train]) & set(sessions[test])


def test_sham_preserves_role_session_and_has_no_fixed_points():
    ids = [f'{role}/{session}/{i}.npy' for role in ('speaker', 'listener')
           for session in ('s1', 's2') for i in range(7)]
    permutation = shuffled_condition_indices(ids)
    assert sorted(permutation) == list(range(len(ids)))
    for i, j in enumerate(permutation):
        assert i != j
        assert ids[i].split('/')[:2] == ids[j].split('/')[:2]
