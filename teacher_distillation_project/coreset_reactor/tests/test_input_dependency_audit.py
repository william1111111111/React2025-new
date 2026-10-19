import numpy as np
import pytest
import torch

from coreset_reactor.input_dependency_audit import donor_maps,block_order,farthest_templates,nearest_train,motion_proxy
from coreset_reactor.fast_frc import candidate_frc


def test_donors_are_equal_length_distinct_and_bijective():
    sessions=['a','a','b','b','a','c'];lengths=[750,750,750,750,600,750]
    within,across,cohort=donor_maps(sessions,lengths)
    assert cohort==[0,1,2,3]
    assert set(within)==set(across)==set(cohort)
    assert set(within.values())==set(across.values())==set(cohort)
    for i in cohort:
        assert within[i]!=i and sessions[i]==sessions[within[i]]
        assert sessions[i]!=sessions[across[i]]
        assert lengths[i]==lengths[within[i]]==lengths[across[i]]==750
    assert (within,across,cohort)==donor_maps(sessions,lengths)


def test_impossible_cross_session_bijection_rejected():
    with pytest.raises(ValueError):donor_maps(['a']*4+['b']*2,[750]*6)


def test_block_shuffle_preserves_all_frames_and_common_multimodal_order():
    for length in (245,750):
        p=block_order(length,'sample')
        assert sorted(p.tolist())==list(range(length))
        assert not torch.equal(p,torch.arange(length))
        assert torch.equal(p,block_order(length,'sample'))


def test_template_selection_is_unique_and_deterministic():
    x=np.random.default_rng(5).normal(size=(30,50))
    ids=farthest_templates(x)
    assert len(set(ids))==10 and ids==farthest_templates(x)


def test_retrieval_uses_only_train_features_with_equal_group_weight():
    groups=[torch.tensor([[0.,0.],[1.,1.]]),torch.tensor([[2.],[0.]]),torch.tensor([[0.],[0.]])]
    query=[torch.zeros(2),torch.zeros(1),torch.zeros(1)]
    assert nearest_train(groups,query,[torch.ones_like(g[0]) for g in groups])[0]==1


def test_all_session_metric_invariance_is_not_a_conditioning_test():
    g=np.random.default_rng(8)
    gt=g.normal(size=(8,30,25)).astype('float32')
    predictions=[g.normal(size=(10,30,25)).astype('float32') for _ in range(4)]
    scores=[candidate_frc(gt,p).sum() for p in predictions]
    swapped=[candidate_frc(gt,predictions[i]).sum() for i in (2,3,0,1)]
    assert np.mean(scores)==pytest.approx(np.mean(swapped))


def test_motion_proxy_constant_inputs_are_finite():
    result=motion_proxy(torch.zeros(10,245,25),torch.zeros(245,25))
    assert result['peak_corr']==0 and result['zero_lag_corr']==0
