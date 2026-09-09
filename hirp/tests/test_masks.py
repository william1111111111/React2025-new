import pytest
import torch


def test_padding_invariance(model, inputs):
    a = model.sample(**inputs, noise=torch.ones(2, 4, 32), return_aux=True)
    changed = {key: value.clone() for key, value in inputs.items()}
    for key in ('speaker_audio', 'speaker_emotion', 'speaker_3dmm'):
        changed[key][1, 19:] = float('nan')
    b = model.sample(**changed, noise=torch.ones(2, 4, 32), return_aux=True)
    torch.testing.assert_close(a['context'], b['context'], rtol=0, atol=0)
    torch.testing.assert_close(a['predictions'], b['predictions'], rtol=0, atol=0)
    # Appending padding must not change the valid convolution boundary.
    extended = dict(inputs)
    for key in ('speaker_audio', 'speaker_emotion', 'speaker_3dmm'):
        extended[key] = torch.cat((inputs[key], torch.randn(2, 7, inputs[key].shape[-1])), 1)
    c = model.sample(**extended, noise=torch.ones(2, 4, 32), return_aux=True)
    torch.testing.assert_close(a['context'], c['context'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a['predictions'], c['predictions'][:, :, :32], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('lengths', [torch.tensor([0, 19]), torch.tensor([33, 19]), torch.tensor([32., 19.])])
def test_invalid_lengths(model, inputs, lengths):
    with pytest.raises(ValueError):
        model(**dict(inputs, lengths=lengths))
