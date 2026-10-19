"""Shared VA correction and a finite, explicitly GT-dependent diagnostic oracle."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from coreset_reactor import oracle_select as quality


VARIANTS = ('baseline', 'quality_centroid', 'oracle_025', 'oracle_050', 'oracle_100')


def feasible_shift(prediction, desired):
    va = prediction[..., 15:17]
    lower = -1-va.amin(0)
    upper = 1-va.amax(0)
    if (lower > 1e-6).any() or (upper < -1e-6).any():
        raise ValueError('Baseline VA outside valid range')
    return desired.clamp(min=lower, max=upper)


def apply_shift(prediction, shift):
    result = prediction.clone()
    result[..., 15:17] += shift
    return result


def ccc_channels(target, prediction):
    """Differentiable all-candidate/GT CCC; final scoring uses native FRC."""
    tm, pm = target.mean(1), prediction.mean(1)
    tc, pc = target-tm[:, None], prediction-pm[:, None]
    covariance = torch.einsum('ktd,ntd->knd', pc, tc)/target.shape[1]
    denominator = (pc.square().mean(1)[:, None] + tc.square().mean(1)[None]
                   + (pm[:, None]-tm[None]).square() + 1e-8)
    return 2*covariance/denominator


def optimize_frc_shift(prediction, target, device, steps=100, controls=16):
    """32 control parameters at most; local FRC search, not a proven optimum."""
    pred, gt = prediction.to(device=device, dtype=torch.float64), target.to(device=device, dtype=torch.float64)
    with torch.no_grad():
        initial = ccc_channels(gt, pred)
        fixed = (initial[..., :15].sum(-1)+initial[..., 17:].sum(-1))/25
    coefficients = torch.zeros(1, 2, min(controls, pred.shape[1]), device=device,
                               dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=0.03)

    def score():
        desired = F.interpolate(coefficients, size=pred.shape[1], mode='linear', align_corners=True)[0].T
        shift = feasible_shift(pred, desired)
        scores = fixed + ccc_channels(gt[..., 15:17], pred[..., 15:17]+shift).sum(-1)/25
        return scores.max(1).values.sum(), shift

    best_score, best_shift = -math.inf, None
    for step in range(steps+1):
        value, shift = score()
        if float(value.detach()) > best_score:
            best_score = float(value.detach())
            best_shift = shift.detach().cpu().float()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            (-value).backward()
            optimizer.step()
    # Final projection in evaluator float32, rather than independent clipping.
    return feasible_shift(prediction, best_shift), best_score


def _va_quality_worker(task):
    """Same exact Mam recurrence; cache invariant AU/expression costs per slot."""
    from tools.compute_frd_resumable import rolling_dtw
    slot, frames, target_count, variants = task
    targets = quality._SHARED_TARGETS[:target_count, :frames]
    all_pred = quality._SHARED_PREDICTIONS[:variants*10, :frames]
    reference = all_pred[slot]
    groups = [np.ascontiguousarray(targets[..., a:b]) for a, b in [(0, 15), (15, 17), (17, 25)]]
    au = np.full(target_count, np.nan)
    expression = np.full(target_count, np.nan)
    results, calls = [], [0, 0, 0]
    for variant in range(variants):
        pred = all_pred[variant*10+slot]
        if not np.array_equal(pred[:, :15], reference[:, :15]) or not np.array_equal(pred[:, 17:], reference[:, 17:]):
            raise ValueError('Invariant-channel cache cannot score changed AU/expression')
        prediction_groups = [np.ascontiguousarray(pred[:, a:b]) for a, b in [(0, 15), (15, 17), (17, 25)]]
        best = math.inf
        for index in range(target_count):
            if np.isnan(au[index]):
                au[index] = float(rolling_dtw(prediction_groups[0], groups[0][index]))/15
                calls[0] += 1
            total = au[index]
            if total >= best:
                continue
            total += float(rolling_dtw(prediction_groups[1], groups[1][index]))
            calls[1] += 1
            if total >= best:
                continue
            if np.isnan(expression[index]):
                expression[index] = float(rolling_dtw(prediction_groups[2], groups[2][index]))/8
                calls[2] += 1
            total += expression[index]
            best = min(best, total)
        results.append(best)
    return slot, results, calls


class SharedVAQualityPool(quality.SharedQualityPool):
    def __init__(self, workers=10, max_targets=113):
        super().__init__(workers, 10*len(VARIANTS), 750, max_targets=max_targets)

    def score_variants(self, predictions, target):
        if self.closed:
            raise RuntimeError('Pool is closed')
        frames, count = target.shape[1], target.shape[0]
        if len(predictions) > len(VARIANTS) or count > self.targets.shape[0] or frames > 750:
            raise ValueError('Buffers exceeded')
        if any(p.shape != (10, frames, 25) for p in predictions):
            raise ValueError('Exactly ten predictions are required')
        self.predictions[:10*len(predictions), :frames] = torch.cat(predictions).numpy()
        self.targets[:count, :frames] = target.numpy()
        result = self.pool.map(_va_quality_worker,
                               [(k, frames, count, len(predictions)) for k in range(10)], chunksize=1)
        result.sort(key=lambda x: x[0])
        return np.array([x[1] for x in result]).T, np.array([x[2] for x in result]).sum(0)


def strict_choice(frc, frd):
    """Zero-slack, whole-ten-set constraints; baseline is always feasible."""
    feasible = [i for i in range(len(frc)) if frc[i] >= frc[0] and frd[i] <= frd[0]]
    return max(feasible, key=lambda i: (frc[i], -frd[i], -i))
