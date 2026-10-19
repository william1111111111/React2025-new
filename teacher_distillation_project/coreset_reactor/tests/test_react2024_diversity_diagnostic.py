from itertools import combinations
import numpy as np
from coreset_reactor.diagnose_react2024_diversity import improve, search, paired_random_expectation
from coreset_reactor.dataset_gt_oracle import frdiv


def distance(n):
    x=np.random.default_rng(4).normal(size=(n,5))
    return ((x[:,None]-x[None])**2).mean(-1)


def test_search_preserves_anchor_uniqueness_and_never_lowers_original_score():
    d=distance(18)
    original=list(range(10))
    selected,value,_=search(d,original,3,4)
    assert selected[0]==0 and len(set(selected))==10
    assert value>=frdiv(d,original)-1e-12
    assert selected==search(d,original,3,4)[0]
    for pos in range(1,10):
        for new in set(range(18))-set(selected):
            candidate=selected.copy(); candidate[pos]=new
            assert frdiv(d,candidate)<=value+1e-10


def test_random_expectation_matches_enumeration():
    d=distance(7)
    for anchor in (0,3):
        values=[frdiv(d,[anchor,*c]) for c in combinations([i for i in range(7) if i!=anchor],3)]
        np.testing.assert_allclose(paired_random_expectation(d,anchor,4),np.mean(values),atol=1e-12)


def test_entire_pool_is_unchanged():
    d=distance(10)
    assert improve(d,list(range(10)),0)==list(range(10))
    np.testing.assert_allclose(paired_random_expectation(d,0),frdiv(d,list(range(10))),atol=1e-12)


def test_quality_gate_never_trades_quality_for_diversity():
    from coreset_reactor.run_reference_count_ablation import quality_gate
    base={'FRC':.65,'exact_FRD':85.,'FRDiv':.14}
    assert quality_gate({**base,'FRDiv':.15},base)
    assert not quality_gate({**base,'FRDiv':.15,'FRC':.6499},base)
    assert not quality_gate({**base,'FRDiv':.15,'exact_FRD':85.001},base)
    assert not quality_gate(base,base)
