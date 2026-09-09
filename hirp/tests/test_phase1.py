import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from hirp.paired_data import PairedReactionDataset, paired_model_inputs, pair_valid_mask
from hirp.phase1_diagnostics import prior_mode, prior_statistics, evaluate_diagnostics
from hirp.losses import paired_energy_score


@pytest.fixture
def paired_fixture(tmp_path):
    norm = tmp_path / 'external' / 'FaceVerse'
    norm.mkdir(parents=True)
    np.save(norm / 'mean_face.npy', np.ones(58, dtype=np.float32))
    np.save(norm / 'std_face.npy', np.full(58, 2, dtype=np.float32))
    data = tmp_path / 'data'
    for name in ('one', 'two'):
        for folder, role, length, width in [('audio-features', 'speaker', 11, 768),
            ('facial-attributes', 'speaker', 13, 25), ('coefficients', 'speaker', 12, 58),
            ('facial-attributes', 'listener', 5, 25)]:
            path = data / 'train' / folder / role / 'session0' / f'{name}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (length, 1, width) if folder == 'coefficients' else (length, width)
            np.save(path, np.random.default_rng(42).normal(size=shape).astype(np.float32))
    dataset = PairedReactionDataset(data, 'train', clip_length=16, crop_mode='start')
    return dataset, next(iter(DataLoader(dataset, batch_size=2)))


def test_distinct_lengths_normalization(paired_fixture):
    dataset, batch = paired_fixture
    assert batch['source_lengths'].tolist() == [11, 11]
    assert batch['pair_lengths'].tolist() == [5, 5]
    assert paired_model_inputs(batch)['lengths'] is batch['source_lengths']
    assert pair_valid_mask(batch).sum(1).tolist() == [5, 5]
    assert 'length' not in batch and 'lengths' not in batch
    assert not batch['speaker_audio'][:, 11:].any()
    assert not batch['paired_target'][:, 5:].any()
    original = torch.from_numpy(np.load(dataset.directory / 'coefficients/speaker/session0/one.npy'))[:, 0]
    torch.testing.assert_close(batch['speaker_3dmm'][0, :11], (original[:11]-1)/2)


def test_target_padding_and_source_tail(model, paired_fixture):
    _, batch = paired_fixture
    noise = torch.randn(2, 4, 32)
    before = model.sample(**paired_model_inputs(batch), noise=noise, return_aux=True)
    batch['paired_target'][:, 5:] = float('nan')
    after = model.sample(**paired_model_inputs(batch), noise=noise, return_aux=True)
    for key in ('context', 'predictions'):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    loss = paired_energy_score(after['predictions'], batch['paired_target'], pair_valid_mask(batch))
    assert torch.isfinite(loss)
    batch['speaker_audio'][:, 5:11] += torch.randn_like(batch['speaker_audio'][:, 5:11])
    changed = model.sample(**paired_model_inputs(batch), noise=noise, return_aux=True)
    assert (changed['context'] - before['context']).abs().max() > 1e-6
    # Only positions beyond source_lengths are irrelevant to the encoder.
    batch['speaker_audio'][:, 11:] = float('nan')
    padded = model.sample(**paired_model_inputs(batch), noise=noise, return_aux=True)
    torch.testing.assert_close(changed['context'], padded['context'], atol=0, rtol=0)


def test_es_details_exact():
    p = torch.randn(2, 4, 9, 25, requires_grad=True)
    y = torch.randn(2, 9, 25)
    mask = torch.arange(9)[None] < torch.tensor([9, 5])[:, None]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        scalar = paired_energy_score(p, y, mask)
        details = paired_energy_score(p, y, mask, return_details=True)
    assert details.keys() == {'loss', 'cross_distance', 'self_distance'}
    assert torch.equal(scalar, details['loss'])
    assert all(value.dtype == torch.float32 for value in details.values())
    torch.testing.assert_close(scalar, details['cross_distance'] - .5*details['self_distance'])
    g1 = torch.autograd.grad(scalar, p, retain_graph=True)[0]
    g2 = torch.autograd.grad(details['loss'], p)[0]
    assert torch.equal(g1, g2)


def test_prior_statistics_finite_singleton():
    stats = prior_statistics(torch.zeros(1, 32), torch.linspace(-4, 2, 32)[None])
    assert all(np.isfinite(value) for value in stats.values())
    assert stats['latent_mu_std_across_batch'] == 0
    assert stats['log_sigma_lower_clamp_fraction'] == 1/32
    assert stats['log_sigma_upper_clamp_fraction'] == 1/32


def test_diagnostics_noise_reproducibility(model, paired_fixture):
    _, batch = paired_fixture
    noise = torch.randn(1, 4, 32).expand(2, -1, -1)
    a = evaluate_diagnostics(model, batch, noise)
    b = evaluate_diagnostics(model, batch, noise)
    assert a == b
    assert all(np.isfinite(value) for value in a.values())
    assert a['same_session_pair_count'] == 1
    assert a['mean_pairwise_reaction_distance'] > 1e-4


def test_standard_normal_ablation(model, paired_fixture):
    _, batch = paired_fixture
    noise = torch.randn(1, 4, 32).expand(2, -1, -1)
    keys = list(model.state_dict())
    with prior_mode(model, 'A0'):
        aux = model(**paired_model_inputs(batch), noise=noise, return_aux=True)
        assert torch.equal(aux['latent_z'], noise)
        assert not aux['latent_mu'].any() and not aux['latent_log_sigma'].any()
        loss = paired_energy_score(aux['predictions'], batch['paired_target'], pair_valid_mask(batch))
        loss.backward()
    assert model.output_head.stochastic.weight.grad.norm() > 0
    assert model.prior.mu.weight.grad is None
    assert list(model.state_dict()) == keys
    restored = model.sample(**paired_model_inputs(batch), noise=noise, return_aux=True)
    assert restored['latent_mu'].abs().sum() > 0
    stats = evaluate_diagnostics(model, batch, noise, 'A0')
    assert stats['sigma_mean'] == 1 and stats['same_session_mu_distance'] == 0
    assert 'target' not in inspect.signature(model.forward).parameters


def test_real_dataloader_contract(model):
    root = Path(__file__).resolve().parents[2]
    if not (root / 'data/train/facial-attributes/speaker').is_dir():
        pytest.skip('real REACT features unavailable')
    dataset = PairedReactionDataset(root / 'data', 'train', clip_length=128)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    for key, width in [('speaker_audio', 768), ('speaker_emotion', 25),
                       ('speaker_3dmm', 58), ('paired_target', 25)]:
        assert batch[key].shape == (2, 128, width)
        assert torch.isfinite(batch[key]).all()
    assert (batch['pair_lengths'] <= batch['source_lengths']).all()
    assert len(batch['clip_id']) == len(batch['session_id']) == 2
    tail = PairedReactionDataset(root / 'data', 'train', clip_length=128, crop_mode='tail')
    short = next(iter(DataLoader(tail, batch_size=2)))
    assert short['source_lengths'].tolist() == [64, 64]
    noise = torch.randn(2, 2, 32)
    first = model.sample(**paired_model_inputs(short), sample_count=2, noise=noise, return_aux=True)
    for key in ('speaker_audio', 'speaker_emotion', 'speaker_3dmm'):
        short[key][:, 64:] = 1e6
    second = model.sample(**paired_model_inputs(short), sample_count=2, noise=noise, return_aux=True)
    torch.testing.assert_close(first['context'], second['context'], atol=0, rtol=0)
    torch.testing.assert_close(first['predictions'], second['predictions'], atol=0, rtol=0)
    print('REAL_BATCH', json.dumps({key: list(value.shape) if torch.is_tensor(value) else value
                                   for key, value in batch.items()}))



def test_real_unequal_pair_lengths(model):
    root = Path(__file__).resolve().parents[2]
    relative = 'speaker/session0/Camera-2024-06-28-164549-164628.npy'
    source_path = root / 'data/train/facial-attributes' / relative
    if not source_path.is_file():
        pytest.skip('specific real unequal-length recording unavailable')
    dataset = PairedReactionDataset(root / 'data', 'train', clip_length=128, crop_mode='tail')
    index = dataset.records.index(source_path)
    batch = torch.utils.data.default_collate([dataset[index]])
    assert batch['source_total_length'].item() == 1018
    assert batch['target_total_length'].item() == 1017
    assert batch['source_lengths'].item() == 64
    assert batch['pair_lengths'].item() == 63
    aux = model.sample(**paired_model_inputs(batch), sample_count=2, return_aux=True)
    assert aux['valid_mask'].sum().item() == 64
    assert pair_valid_mask(batch).sum().item() == 63
    loss = paired_energy_score(aux['predictions'], batch['paired_target'], pair_valid_mask(batch))
    changed = batch['paired_target'].clone()
    changed[:, 63:] = float('nan')
    assert torch.equal(loss, paired_energy_score(aux['predictions'], changed, pair_valid_mask(batch)))
    print('REAL_UNEQUAL_LENGTHS', relative, 'source=64 pair=63 crop_start=954')
