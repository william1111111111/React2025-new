import numpy as np
import torch
from coreset_reactor.teacher_diversity_decomposition import decompose


def test_exact_pairwise_identity():
    p=torch.randn(10,71,25,generator=torch.Generator().manual_seed(99))
    d=decompose(p)
    ref=float(torch.cdist(p.double().flatten(1),p.double().flatten(1)).square().sum()/(10*9*71*25))
    np.testing.assert_allclose(d['FRDiv'],ref,atol=1e-12)
    np.testing.assert_allclose(d['time_mean_component']+d['temporal_residual_component'],ref,atol=1e-12)
    np.testing.assert_allclose(sum(d[k] for k in ('AU_weighted_component','VA_weighted_component','expression_weighted_component')),ref,atol=1e-12)


def test_constant_modes_have_no_temporal_residual_diversity():
    p=torch.arange(10)[:,None,None].expand(10,20,25).float()
    d=decompose(p)
    assert d['temporal_residual_component']==0
    assert d['FRDiv']==d['time_mean_component']
