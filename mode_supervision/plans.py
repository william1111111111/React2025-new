"""Fixed PTS cosine summaries in frozen T0 coordinates, with explicit validity."""
import math
import torch


def basis(pts):
    pts = torch.as_tensor(pts, dtype=torch.float64)
    n = len(pts)
    if n < 1 or not torch.isfinite(pts).all() or (n > 1 and not (pts.diff() > 0).all()):
        raise ValueError('nonempty strictly increasing finite PTS required')
    m = min(4, n)
    # Midpoint phase with measured local endpoint interval; no assumed frame rate.
    dt = pts.diff().median() if n > 1 else pts.new_tensor(1.)
    phase = (pts - pts[0] + dt / 2) / (pts[-1] - pts[0] + dt)
    design = torch.cos(math.pi * phase[:, None] * torch.arange(m, device=pts.device))
    q, r = torch.linalg.qr(design, mode='reduced')
    q = q * torch.where(r.diagonal() < 0, -1., 1.)
    return q


def summarize(u, pts):
    """u [...,n,24] -> [...,4,24], basis mask [4,24]. Differentiable in u."""
    if u.shape[-1] != 24 or u.shape[-2] != len(pts):
        raise ValueError('coordinates/PTS mismatch')
    q = basis(pts).to(u)
    a = torch.einsum('nm,...nd->...md', q, u) / math.sqrt(len(pts))
    a = torch.nn.functional.pad(a, (0, 0, 0, 4 - q.shape[1]))
    mask = torch.arange(4, device=u.device)[:, None].expand(4, 24) < q.shape[1]
    return a, mask


def recording(u, pts, block=750):
    if len(u) != len(pts):
        raise ValueError('trajectory/PTS mismatch')
    pairs = [summarize(u[i:i+block], pts[i:i+block]) for i in range(0, len(u), block)]
    return torch.stack([a.flatten() for a, _ in pairs]), torch.stack([m.flatten() for _, m in pairs])


def mode_loss(actual, desired, mask, scales):
    """Equal AU/VA/expression-coordinate groups; scales fitted only on TRAIN."""
    value = torch.nn.functional.smooth_l1_loss(actual / scales, desired / scales, reduction='none')
    value, mask = value.reshape(*value.shape[:-1], 4, 24), mask.reshape(*mask.shape[:-1], 4, 24)
    groups = []
    for lo, hi in ((0, 15), (15, 17), (17, 24)):
        w = mask[..., lo:hi]
        groups.append((value[..., lo:hi] * w).sum() / w.sum().clamp_min(1))
    return torch.stack(groups).mean()
