import torch
import pytest
from hirp.data_adapter import generator_inputs


def test_shape_domains_aux(model, inputs):
    with torch.no_grad():
        aux = model(**inputs, return_aux=True)
    p = aux['predictions']
    assert p.shape == (2, 4, 32, 25)
    valid = aux['valid_mask'][:, None].expand(2, 4, 32)
    q = p[valid]
    assert ((q[:, :15] >= 0) & (q[:, :15] <= 1)).all()
    assert ((q[:, 15:17] >= -1) & (q[:, 15:17] <= 1)).all()
    torch.testing.assert_close(q[:, 17:].sum(-1), torch.ones_like(q[:, 0]))
    assert torch.count_nonzero(p[~valid]) == 0
    assert aux['context'].shape == (2, 256)
    assert aux['latent_mu'].shape == aux['latent_log_sigma'].shape == (2, 32)
    assert aux['latent_z'].shape == aux['noise'].shape == (2, 4, 32)
    assert aux['latent_log_sigma'].min() >= -4 and aux['latent_log_sigma'].max() <= 2
    torch.testing.assert_close(aux['latent_z'], aux['latent_mu'][:, None] + aux['latent_log_sigma'].exp()[:, None] * aux['noise'])
    print('FORWARD_SHAPE', tuple(p.shape))


@pytest.mark.parametrize('k', [1, 2, 4, 10, 32])
def test_arbitrary_k(model, inputs, k):
    assert model.sample(**inputs, sample_count=k).shape == (2, k, 32, 25)


def test_adapter(inputs):
    batch = dict(inputs)
    batch['length'] = batch.pop('lengths')
    batch['target'] = torch.randn(2, 32, 25)
    adapted = generator_inputs(batch)
    assert adapted.keys() == inputs.keys()
    assert adapted['lengths'] is batch['length']
