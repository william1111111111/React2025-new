import torch

from coreset_reactor.learned_common_va import CommonVACorrector, feasible_batch_shift, apply_batch_shift


def test_shapes_padding_and_geometry():
    torch.manual_seed(4)
    model = CommonVACorrector(dropout=0).eval()
    values = [torch.randn(2, 17, d) for d in (768, 25, 58, 25, 25)]
    lengths = torch.tensor([17, 11])
    out = model(*values, lengths)
    assert out['shift'].shape == (2, 17, 2) and out['gate_logit'].shape == (2,)
    assert torch.equal(out['shift'][1, 11:], torch.zeros_like(out['shift'][1, 11:]))
    prediction = torch.rand(2, 10, 17, 25)
    shift = feasible_batch_shift(prediction, out['shift'])
    corrected = apply_batch_shift(prediction, shift)
    torch.testing.assert_close(corrected-corrected.mean(1, keepdim=True),
                               prediction-prediction.mean(1, keepdim=True), atol=2e-7, rtol=0)
    assert corrected[..., 15:17].abs().max() <= 1+1e-7
    assert torch.equal(corrected[..., :15], prediction[..., :15])
    assert torch.equal(corrected[..., 17:], prediction[..., 17:])
    (out['shift'].square().mean()+out['gate_logit'].square().mean()).backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    for prefix in ('audio.', 'temporal.', 'shift.', 'gate.'):
        assert any(name.startswith(prefix) and value.abs().sum() > 0
                   for name, value in gradients.items())


def test_parameter_budget_is_small():
    count = sum(p.numel() for p in CommonVACorrector().parameters())
    assert 100_000 < count < 300_000
