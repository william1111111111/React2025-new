import torch


def paired_energy_score(predictions, target, valid_mask, channel_scale=None, eps=1e-8, return_details=False):
    """Unbiased off-diagonal paired ES; K=1 explicitly rejected.

    Each distance is RMS over valid time/channel elements, with eps under sqrt.
    Positive fixed channel scales divide errors. Reduction averages examples.
    """
    if predictions.ndim != 4 or predictions.shape[-1] != 25:
        raise ValueError('predictions must be [B,K,T,25]')
    b, k, t, c = predictions.shape
    if k < 2:
        raise ValueError('unbiased self-term requires K >= 2')
    if target.shape != (b, t, c) or valid_mask.shape != (b, t):
        raise ValueError('target or valid_mask shape mismatch')
    if valid_mask.dtype != torch.bool or not valid_mask.any(-1).all() or b == 0:
        raise ValueError('valid_mask must be boolean with nonempty examples')
    if eps <= 0:
        raise ValueError('eps must be positive')
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        p, y = predictions.float(), target.float()
        scale = torch.ones(c, device=p.device) if channel_scale is None else torch.as_tensor(channel_scale, device=p.device, dtype=torch.float32)
        if scale.shape != (c,) or scale.requires_grad or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('channel_scale must be fixed, finite, positive [25]')
        mask = valid_mask[:, None, :, None]
        denom = valid_mask.sum(-1).float()[:, None] * c

        def distance(a, other):
            delta = (a - other).masked_fill(~mask, 0) / scale
            return (delta.square().sum((-1, -2)) / denom + eps).sqrt()

        cross = distance(p, y[:, None]).mean(1)
        # Stream pairs to avoid a [B,K,K,T,C] allocation.
        pair_sum = p.new_zeros(b)
        for i in range(k-1):
            pair_sum = pair_sum + distance(p[:, i:i+1], p[:, i+1:]).sum(1)
        self_term = pair_sum * (2.0 / (k * (k-1)))
        loss = (cross - 0.5 * self_term).mean()
        if return_details:
            return dict(loss=loss, cross_distance=cross.mean(), self_distance=self_term.mean())
        return loss
