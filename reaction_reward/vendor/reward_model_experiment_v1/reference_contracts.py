"""Reference contracts only. No REACT files, model weights, or training are loaded."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreferenceEvidence:
    split: str
    source_group: str
    a_groups: tuple[str, ...]
    b_groups: tuple[str, ...]
    family: str
    head: str
    prefer_a: Optional[float]  # None means UNKNOWN, not 0.5.
    confidence_weight: float

    def validate(self, assignments: dict[str, str]) -> None:
        if self.split not in {'RM_fit', 'RM_cal', 'RM_audit'}:
            raise ValueError('Unknown split')
        for group in (self.source_group, *self.a_groups, *self.b_groups):
            if assignments.get(group) != self.split:
                raise ValueError(f'Cross-split or unknown provenance: {group}')
        if self.family == 'weak_same_session' and self.head != 'context':
            raise ValueError('Same-session membership does not establish temporal correspondence')
        if self.prefer_a is not None and self.prefer_a not in (0., 0.5, 1.):
            raise ValueError('Preference must be 0, 0.5, 1, or UNKNOWN')
        if not np.isfinite(self.confidence_weight) or self.confidence_weight < 0:
            raise ValueError('Invalid weight')
        if self.prefer_a == 0.5 and self.family != 'independently_verified_tie':
            raise ValueError('Insufficient or conflicting evidence is not an automatic tie')


def preference_loss(score_a: torch.Tensor, score_b: torch.Tensor,
                    prefer_a: torch.Tensor, known: torch.Tensor,
                    weight: torch.Tensor) -> torch.Tensor:
    """Within one evidence family; callers combine family-normalized losses."""
    if not all(x.shape == score_a.shape for x in (score_b, prefer_a, known, weight)):
        raise ValueError('Shape mismatch')
    mask = known.to(dtype=torch.bool, device=score_a.device)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Weights must be finite and nonnegative')
    if mask.any() and not ((prefer_a[mask] >= 0) & (prefer_a[mask] <= 1)).all():
        raise ValueError('Known preferences must lie in [0,1]')
    target = torch.where(mask, prefer_a, torch.zeros_like(prefer_a))
    losses = F.binary_cross_entropy_with_logits(score_a - score_b, target,
                                                reduction='none')
    w = weight.to(score_a) * mask.to(score_a.dtype)
    return (losses * w).sum() / w.sum().clamp_min(1e-12)


def pareto_preferences(ccc: Sequence[float], dtw: Sequence[float],
                       c_margin: float, d_margin: float) -> np.ndarray:
    """P[i,j]=1 only for strict two-axis superiority; nan means abstention."""
    c, d = np.asarray(ccc, dtype=float), np.asarray(dtw, dtype=float)
    if c.ndim != 1 or c.shape != d.shape or not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError('Finite vectors of equal size required')
    if c_margin <= 0 or d_margin <= 0:
        raise ValueError('Use positive preregistered margins')
    out = np.full((len(c), len(c)), np.nan)
    wins = (c[:, None] >= c[None, :] + c_margin) & (d[:, None] <= d[None, :] - d_margin)
    out[wins] = 1.
    out[wins.T] = 0.
    return out


def contiguous_shift_pair(y: np.ndarray, start: int, length: int,
                          offset: int) -> tuple[np.ndarray, np.ndarray]:
    """A utility to obtain real contiguous windows, not to label them incorrect."""
    if length < 2 or start < 0 or start + length > len(y):
        raise ValueError('Invalid reference interval')
    shifted = start + offset
    if shifted < 0 or shifted + length > len(y):
        raise ValueError('No circular wrap, synthetic padding, or silent truncation')
    return y[start:start + length].copy(), y[shifted:shifted + length].copy()


def anchored_score(score: np.ndarray, reference: np.ndarray,
                   temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive fixed calibration temperature required')
    return (np.asarray(score) - np.asarray(reference).mean()) / temperature
