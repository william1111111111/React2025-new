import numpy as np

from coreset_reactor.supported_teacher_manifest import supported_assignment


def test_assignment_preserves_pair_distinctness_and_input_order_invariance():
    rng = np.random.default_rng(10)
    values, centers = rng.normal(size=(24, 6)), rng.normal(size=(9, 6))
    ids = [f'speaker/session0/{i}.npy' for i in range(24)]
    pairs = list(range(24))
    a, _ = supported_assignment(values, centers, pairs, ids)
    b, _ = supported_assignment(values, centers, pairs[::-1], ids[::-1])
    assert a == b
    for i, key in enumerate(ids):
        assert a[key][0] == i
        assert len(a[key]) == len(set(a[key])) == 10
        assert all(0 <= x < 24 for x in a[key])


def test_reuse_penalty_reduces_repeated_prototype_assignments():
    values = np.arange(30, dtype=float)[:, None]/30
    centers = np.zeros((9, 1))
    ids = [f'speaker/session0/{i}.npy' for i in range(30)]
    pairs = list(range(30))
    _, plain = supported_assignment(values, centers, pairs, ids, reuse_weight=0)
    _, penalized = supported_assignment(values, centers, pairs, ids, reuse_weight=10)
    assert penalized['max_target_uses_per_slot'] < plain['max_target_uses_per_slot']
