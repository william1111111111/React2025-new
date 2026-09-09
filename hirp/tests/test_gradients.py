import torch
from hirp.losses import paired_energy_score


def test_gradients(model, inputs):
    model.train()
    aux = model(**inputs, return_aux=True)
    loss = paired_energy_score(aux['predictions'], torch.randn(2, 32, 25), aux['valid_mask'])
    loss.backward()
    modules = {'audio_stem': model.stems.audio, 'speaker_encoder': model.encoder,
               'prior_mu': model.prior.mu, 'prior_log_sigma': model.prior.log_sigma,
               'decoder': model.decoder, 'output_head': model.output_head}
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        norm = torch.stack([g.float().square().sum() for g in grads]).sum().sqrt().item()
        assert norm > 0
        print('GRADIENT_NORM', name, norm)
