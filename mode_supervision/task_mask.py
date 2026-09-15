"""Retain one-frame blocks for FM/mode; CCC/task needs >=2 common frames."""
import torch
from reaction_flow.task_dynamics_train import task_parts


def masked_task_parts(pred, batch):
    valid = (batch['source_lengths'] >= 2) & (batch['pair_lengths'] >= 2) & (batch['_target_lengths'] >= 2).all(-1)
    count = int(valid.sum())
    if count == len(pred):
        loss, parts = task_parts(pred, batch)
    elif count:
        selected = {k: v[valid] if isinstance(v, torch.Tensor) and v.ndim and len(v) == len(pred) else v for k, v in batch.items()}
        loss, parts = task_parts(pred[valid], selected)
        # Keep each supported source's original 1/B contribution.
        loss = loss * count / len(pred)
    else:
        loss, parts = pred.sum() * 0, {}
    return loss, {**parts, 'task_supported_samples': count, 'task_unsupported_samples': len(pred)-count, 'task_reduction': 'supported sum / original batch size'}
