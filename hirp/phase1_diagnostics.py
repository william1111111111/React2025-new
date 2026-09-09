"""Read-only mechanism probes and an experiment-only standard-prior ablation."""
from contextlib import contextmanager
import torch
from .paired_data import paired_model_inputs, pair_valid_mask
from .losses import paired_energy_score


@contextmanager
def prior_mode(model, mode='A1'):
    """A0 retains the exact parameter structure but uses mu=0, log_sigma=0.

    Prior parameters receive no gradient in A0; no parameter is removed/added.
    A_global passes zero context and learns only prior biases.
    A1 is the unchanged conditional prior. Hooks are removed on all exit paths.
    """
    if mode not in ('A0', 'A_global', 'A1'):
        raise ValueError('prior mode must be A0, A_global or A1')
    handle = None
    frozen = []
    if mode == 'A0':
        handle = model.prior.register_forward_hook(
            lambda module, inputs, output: (torch.zeros_like(output[0]), torch.zeros_like(output[1])))
    elif mode == 'A_global':
        # Retain identical parameters, but only biases may learn. Disabling
        # weight gradients also prevents AdamW from decaying unused weights.
        for layer in (model.prior.mu, model.prior.log_sigma):
            frozen.append((layer.weight, layer.weight.requires_grad))
            layer.weight.requires_grad_(False)
        handle = model.prior.register_forward_pre_hook(
            lambda module, inputs: (torch.zeros_like(inputs[0]),))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()
        for parameter, required in frozen:
            parameter.requires_grad_(required)


def gradient_norm(module):
    gradients = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    if not gradients:
        return 0.0
    return torch.stack([g.square().sum() for g in gradients]).sum().sqrt().item()


def prior_statistics(mu, log_sigma):
    mu, log_sigma = mu.float(), log_sigma.float()
    sigma = log_sigma.exp()
    return dict(latent_mu_mean=mu.mean().item(),
                latent_mu_std_across_batch=mu.std(0, unbiased=False).mean().item(),
                sigma_mean=sigma.mean().item(), sigma_std=sigma.std(unbiased=False).item(),
                sigma_min=sigma.min().item(), sigma_max=sigma.max().item(),
                log_sigma_lower_clamp_fraction=(log_sigma <= -4).float().mean().item(),
                log_sigma_upper_clamp_fraction=(log_sigma >= 2).float().mean().item())


def _rms(delta, mask):
    # delta is [K,T,C]; valid temporal positions only, same normalized norm as ES.
    delta = delta.float().masked_fill(~mask[None, :, None], 0)
    return (delta.square().sum((-1, -2)) / (mask.sum() * delta.shape[-1]) + 1e-8).sqrt()


@torch.no_grad()
def evaluate_diagnostics(model, batch, noise, mode='A1'):
    """Use caller-owned explicit noise; same bank across inputs enables comparisons.

    Noise usage and residual metrics use source_lengths. ES uses pair_lengths.
    Same-session distances use the intersection of the two valid source masks.
    tanh saturation = fraction with abs(tanh(stochastic_raw)) >= 0.99.
    Coordinates of mu/sigma are mechanism probes, not semantic representations.
    """
    if noise.ndim != 3 or noise.shape[1] < 2:
        raise ValueError('evaluate_diagnostics requires noise [B,K,D] with K >= 2')
    modes = [(module, module.training) for module in model.modules()]
    captured = []
    handle = model.output_head.stochastic.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach()))
    try:
        model.eval()
        with prior_mode(model, mode):
            aux = model(**paired_model_inputs(batch), sample_count=noise.shape[1],
                        noise=noise, return_aux=True)
    finally:
        handle.remove()
        for module, training in modes:
            module.training = training
    p, mask = aux['predictions'], aux['valid_mask']
    details = paired_energy_score(p, batch['paired_target'], pair_valid_mask(batch), return_details=True)
    result = {key: value.item() for key, value in details.items()}
    result.update(prior_statistics(aux['latent_mu'], aux['latent_log_sigma']))
    pairs = []
    for b in range(p.shape[0]):
        for i in range(p.shape[1]):
            for j in range(i+1, p.shape[1]):
                pairs.append(_rms(p[b, i:i+1] - p[b, j:j+1], mask[b]).mean())
    result['mean_pairwise_reaction_distance'] = torch.stack(pairs).mean().item()
    result['mean_absolute_prediction_std_across_samples'] = p.float().std(1, unbiased=False)[mask].abs().mean().item()
    raw = captured[0].float()[mask[:, None].expand(p.shape[:3])]
    result.update(stochastic_raw_abs_mean=raw.abs().mean().item(),
                  stochastic_residual_abs_mean=raw.tanh().abs().mean().item(),
                  tanh_saturation_fraction=(raw.tanh().abs() >= .99).float().mean().item())
    mu_distances, sigma_distances, output_distances = [], [], []
    for i in range(p.shape[0]):
        for j in range(i+1, p.shape[0]):
            if batch['session_id'][i] != batch['session_id'][j]:
                continue
            if not torch.equal(noise[i], noise[j]):
                raise ValueError('same-session diagnostics require the same explicit noise bank')
            common = mask[i] & mask[j]
            mu_distances.append((aux['latent_mu'][i]-aux['latent_mu'][j]).float().norm())
            sigma_distances.append((aux['latent_log_sigma'][i].exp()-aux['latent_log_sigma'][j].exp()).float().norm())
            output_distances.append(_rms(p[i]-p[j], common).mean())
    result['same_session_pair_count'] = len(mu_distances)
    for name, values in [('same_session_mu_distance', mu_distances),
                         ('same_session_sigma_distance', sigma_distances),
                         ('same_session_fixed_noise_output_distance', output_distances)]:
        result[name] = torch.stack(values).mean().item() if values else None
    return result
