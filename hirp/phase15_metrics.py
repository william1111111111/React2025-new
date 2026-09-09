"""Evaluation-only channel metrics and same-session real trajectory references.

Distances are RMS Euclidean with eps=1e-8, as in paired ES. All summaries are
per-example first, then macro-averaged. Velocity is a one-frame difference
(no assumed fps). Boundary tolerance is 0.01; VA range is [-1,1], others [0,1].
"""
from collections import defaultdict
import numpy as np
import torch

GROUPS = {'overall': slice(0,25), 'AU': slice(0,15), 'VA': slice(15,17), 'expression': slice(17,25)}


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return dict(count=0, mean=None, std=None, p05=None, p50=None, p95=None)
    return dict(count=int(array.size), mean=float(array.mean()), std=float(array.std()),
                p05=float(np.quantile(array,.05)), p50=float(np.quantile(array,.5)), p95=float(np.quantile(array,.95)))


def trajectory_summary(value, name):
    """value [K,valid_T,C], detached float32; no padded frames included."""
    if value.ndim != 3 or value.shape[1] < 1:
        raise ValueError('nonempty [K,T,C] required')
    low = -1 if name == 'VA' else 0
    if name == 'overall':
        lower = torch.zeros(25, device=value.device)
        lower[15:17] = -1
        boundary = (value <= lower+.01) | (value >= .99)
    else:
        boundary = (value <= low+.01) | (value >= .99)
    result = dict(boundary_fraction=boundary.float().mean().item(),
                  temporal_mean=value.mean().item(), temporal_std=value.std(1, unbiased=False).mean().item())
    if value.shape[1] > 1:
        velocity = value[:, 1:] - value[:, :-1]
        result.update(velocity_abs_mean=velocity.abs().mean().item(),
                      velocity_rms=velocity.square().mean().sqrt().item())
    else:
        result.update(velocity_abs_mean=None, velocity_rms=None)
    if name in ('expression','overall'):
        probabilities = value if name == 'expression' else value[...,17:25]
        result['expression_entropy'] = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean().item()
    return result


@torch.no_grad()
def channel_metrics(predictions, target, lengths):
    if predictions.ndim != 4 or predictions.shape[1] < 2:
        raise ValueError('channel_metrics requires [B,K,T,25], K >= 2')
    p, y = predictions.float(), target.float()
    k = p.shape[1]
    ij = torch.triu_indices(k,k,offset=1,device=p.device)
    records = []
    for b in range(p.shape[0]):
        n = int(lengths[b])
        if not 1 <= n <= p.shape[2]:
            raise ValueError('invalid evaluation length')
        result = {}
        for name, channels in GROUPS.items():
            q, paired = p[b,:,:n,channels], y[b,:n,channels]
            cross = ((q-paired).square().mean((-1,-2))+1e-8).sqrt().mean()
            self_distance = ((q[ij[0]]-q[ij[1]]).square().mean((-1,-2))+1e-8).sqrt().mean()
            result[name] = dict(cross_distance=cross.item(), self_distance=self_distance.item(),
                                ES=(cross-.5*self_distance).item(),
                                prediction_std=q.std(0,unbiased=False).mean().item(),
                                **trajectory_summary(q,name))
        records.append(result)
    return records


def aggregate_channels(records):
    return {name: {key: float(np.mean([r[name][key] for r in records if r[name][key] is not None]))
                   if any(r[name][key] is not None for r in records) else None
                   for key in records[0][name]} for name in GROUPS}


@torch.no_grad()
def real_real_reference(batch):
    """All unordered pairs among selected same-session real paired targets.

    Crops are aligned by relative frame position only; these are different
    inputs, NOT samples of a known common conditional distribution. The
    reference is descriptive and is never called by the training loop.
    """
    values, pairs = [], []
    y = batch['paired_target'].float()
    for i in range(len(y)):
        length = int(batch['pair_lengths'][i])
        values.append(dict(clip_id=batch['clip_id'][i], session_id=batch['session_id'][i],
                           channels={name: trajectory_summary(y[i:i+1,:length,channels],name)
                                     for name,channels in GROUPS.items()}))
        for j in range(i+1,len(y)):
            if batch['session_id'][i] != batch['session_id'][j]:
                continue
            n = min(length,int(batch['pair_lengths'][j]))
            distances, summary_distances = {}, {}
            for name, channels in GROUPS.items():
                a,b = y[i,:n,channels],y[j,:n,channels]
                distances[name] = ((a-b).square().mean()+1e-8).sqrt().item()
                summary_distances[name] = dict(
                    temporal_mean_distance=((a.mean(0)-b.mean(0)).square().mean()+1e-8).sqrt().item(),
                    temporal_std_distance=((a.std(0,unbiased=False)-b.std(0,unbiased=False)).square().mean()+1e-8).sqrt().item())
            pairs.append(dict(first=batch['clip_id'][i],second=batch['clip_id'][j],
                              session_id=batch['session_id'][i],common_length=n,
                              distances=distances,summary_distances=summary_distances))
    aggregate={}
    for name in GROUPS:
        aggregate[name]=dict(distance=distribution([p['distances'][name] for p in pairs]),
            summaries={key:distribution([v['channels'][name][key] for v in values if v['channels'][name][key] is not None])
                       for key in values[0]['channels'][name]},
            summary_distances={key:distribution([p['summary_distances'][name][key] for p in pairs])
                               for key in ('temporal_mean_distance','temporal_std_distance')})
    return dict(recording_count=len(values),same_session_pair_count=len(pairs),
                aggregate=aggregate,per_recording=values,pairs=pairs)
