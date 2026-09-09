import inspect
from pathlib import Path
import pytest
import torch


def test_reproducibility_prefix(model, inputs):
    noise = torch.randn(2, 10, 32)
    a = model.sample(**inputs, sample_count=4, noise=noise[:, :4])
    b = model.sample(**inputs, sample_count=10, noise=noise)
    c = model.sample(**inputs, sample_count=4, noise=noise[:, :4])
    torch.testing.assert_close(a, c, rtol=0, atol=0)
    error = (a-b[:, :4]).abs().max().item()
    print('K_PREFIX_MAX_ERROR', error)
    torch.testing.assert_close(a, b[:, :4], rtol=1e-5, atol=1e-6)
    perm = torch.randperm(10)
    d = model.sample(**inputs, sample_count=10, noise=noise[:, perm])
    torch.testing.assert_close(d, b[:, perm], rtol=1e-5, atol=1e-6)


def test_sensitivities(model, inputs):
    noise = torch.zeros(2, 4, 32)
    a = model.sample(**inputs, noise=noise, return_aux=True)
    b = model.sample(**inputs, noise=noise+5)
    assert (a['predictions']-b).abs().max() > 1e-6
    changed = dict(inputs, speaker_audio=inputs['speaker_audio'] + torch.randn_like(inputs['speaker_audio']))
    c = model.sample(**changed, noise=noise, return_aux=True)
    for key in ('predictions', 'latent_mu', 'latent_log_sigma'):
        assert (a[key]-c[key]).abs().max() > 1e-6


def test_sample_wrapper(model, inputs):
    model.train()
    p = model.sample(**inputs)
    assert not p.requires_grad and model.training
    assert all(m.training for m in model.modules())


def test_api_and_independence(model):
    assert 'target' not in inspect.signature(model.forward).parameters
    assert 'target' not in inspect.getsource(model.forward)
    for path in (Path(__file__).parents[1] / 'model').glob('*.py'):
        source = path.read_text()
        assert 'regnn' not in source and 'torch.load' not in source
    assert not any('sample_count' in key or 'query_index' in key for key in model.state_dict())


def test_explicit_noise_no_rng(model, inputs):
    noise = torch.randn(2, 4, 32)
    before = torch.random.get_rng_state().clone()
    aux = model.sample(**inputs, noise=noise, return_aux=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert aux['noise'] is noise


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_bad_k(model, inputs, k):
    with pytest.raises(ValueError):
        model(**inputs, sample_count=k)


@pytest.mark.parametrize('modality', ['speaker_audio', 'speaker_emotion', 'speaker_3dmm'])
def test_each_modality_conditions_prior_and_output(model, inputs, modality):
    noise = torch.randn(2, 4, 32)
    original = model.sample(**inputs, noise=noise, return_aux=True)
    changed = dict(inputs)
    changed[modality] = inputs[modality] + torch.randn_like(inputs[modality])
    perturbed = model.sample(**changed, noise=noise, return_aux=True)
    for key in ('predictions', 'latent_mu', 'latent_log_sigma'):
        assert (original[key] - perturbed[key]).abs().max() > 1e-6


def test_prior_clamp(model):
    with torch.no_grad():
        model.prior.log_sigma.weight.zero_()
        model.prior.log_sigma.bias[:16].fill_(-100)
        model.prior.log_sigma.bias[16:].fill_(100)
        _, log_sigma = model.prior(torch.randn(2, 256))
    assert torch.equal(log_sigma[:, :16], torch.full((2, 16), -4.))
    assert torch.equal(log_sigma[:, 16:], torch.full((2, 16), 2.))


@pytest.mark.parametrize('noise', [torch.randn(2, 3, 32), torch.randn(2, 4, 31),
                                   torch.randn(2, 4, 32, dtype=torch.float64)])
def test_invalid_explicit_noise(model, inputs, noise):
    with pytest.raises(ValueError, match='noise'):
        model.sample(**inputs, noise=noise)


def test_sample_restores_mixed_modes_after_exception(model, inputs):
    model.train()
    model.encoder.eval()
    before = [module.training for module in model.modules()]
    with pytest.raises(ValueError):
        model.sample(**inputs, sample_count=0)
    assert [module.training for module in model.modules()] == before
