"""Reference tensor operations only; this file does not load/train BERT or REACT."""
from __future__ import annotations

import math
import torch
from torch import Tensor


def masked_content_mean(hidden: Tensor, attention_mask: Tensor, special_mask: Tensor) -> Tensor:
    """Mean of contextual content tokens, excluding padding and special tokens.

    An empty content sequence is a caller-visible error. The data loader should
    represent it as NULL rather than silently creating an arbitrary embedding.
    """
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or special_mask.shape != hidden.shape[:2]:
        raise ValueError('Expected hidden[B,L,D] and masks[B,L]')
    keep = attention_mask.to(device=hidden.device, dtype=torch.bool) & ~special_mask.to(device=hidden.device, dtype=torch.bool)
    counts = keep.sum(dim=1)
    if (counts == 0).any():
        raise ValueError('No content tokens; caller must use NULL')
    # masked_fill also removes arbitrary padding values before the reduction.
    selected = hidden.masked_fill(~keep[..., None], 0)
    return selected.sum(dim=1) / counts[:, None].to(hidden.dtype)


def event_time_bias(
    frame_seconds: Tensor,
    intervals_seconds: Tensor,
    known_time: Tensor,
    *,
    sigma_seconds: float = 1.0,
    post_slack_seconds: float = 2.0,
    minimum_bias: float = -8.0,
) -> Tensor:
    """Return a soft additive bias[B,T,J] in a common measured time coordinate.

    This is NOT a causal mask. Unknown time receives zero bias, not guessed
    timestamps. Padding and the NULL column must be handled by the attention
    caller. The constants are proposed experimental priors, not fitted facts.
    """
    if frame_seconds.ndim != 2 or intervals_seconds.ndim != 3 or intervals_seconds.shape[-1] != 2:
        raise ValueError('Expected frames[B,T], intervals[B,J,2]')
    if intervals_seconds.shape[0] != frame_seconds.shape[0] or known_time.shape != intervals_seconds.shape[:2]:
        raise ValueError('Batch or event-mask shape mismatch')
    if not (math.isfinite(sigma_seconds) and sigma_seconds > 0):
        raise ValueError('sigma_seconds must be finite and positive')
    if not (math.isfinite(post_slack_seconds) and post_slack_seconds >= 0):
        raise ValueError('post_slack_seconds must be finite and nonnegative')
    if not (math.isfinite(minimum_bias) and minimum_bias < 0):
        raise ValueError('minimum_bias must be finite and negative')
    if not torch.isfinite(frame_seconds).all():
        raise ValueError('Frame timestamps must be finite')
    known = known_time.to(device=frame_seconds.device, dtype=torch.bool)
    bounds = intervals_seconds.to(device=frame_seconds.device, dtype=frame_seconds.dtype)
    if not torch.isfinite(bounds[known]).all() or (bounds[..., 1][known] <= bounds[..., 0][known]).any():
        raise ValueError('Known event intervals must be finite and positive length')
    bounds = torch.where(known[..., None], bounds, torch.zeros_like(bounds))
    t = frame_seconds[..., None]
    start = bounds[:, None, :, 0]
    end = bounds[:, None, :, 1] + post_slack_seconds
    distance = torch.maximum((start - t).clamp_min(0), (t - end).clamp_min(0))
    bias = (-0.5 * (distance / sigma_seconds).square()).clamp(min=minimum_bias, max=0.0)
    return torch.where(known[:, None, :], bias, torch.zeros_like(bias))
