"""Inference-safe common VA corrector; no listener-GT input or metric access."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2*dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1), nn.GroupNorm(8, channels))

    def forward(self, value):
        return F.gelu(value+self.net(value))


class CommonVACorrector(nn.Module):
    def __init__(self, hidden=64, dropout=.1, max_shift=.35):
        super().__init__()
        self.max_shift = float(max_shift)
        self.audio = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32), nn.GELU())
        self.emotion = nn.Sequential(nn.LayerNorm(25), nn.Linear(25, 16), nn.GELU())
        self.face = nn.Sequential(nn.LayerNorm(58), nn.Linear(58, 16), nn.GELU())
        self.mean = nn.Sequential(nn.LayerNorm(25), nn.Linear(25, 16), nn.GELU())
        self.std = nn.Sequential(nn.LayerNorm(25), nn.Linear(25, 16), nn.GELU())
        self.input = nn.Conv1d(96, hidden, 1)
        self.temporal = nn.Sequential(*[ResidualTemporalBlock(hidden, d, dropout) for d in (1, 2, 4)])
        self.shift = nn.Conv1d(hidden, 2, 1)
        self.gate = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, audio, emotion, face, teacher_mean, teacher_std, lengths):
        prefix = audio.shape[:2]
        if any(x.shape[:2] != prefix for x in (emotion, face, teacher_mean, teacher_std)):
            raise ValueError('All temporal inputs must share [B,T]')
        lengths = torch.as_tensor(lengths, device=audio.device, dtype=torch.long).clamp(1, audio.shape[1])
        valid = torch.arange(audio.shape[1], device=audio.device)[None] < lengths[:, None]
        x = torch.cat((self.audio(audio), self.emotion(emotion), self.face(face),
                       self.mean(teacher_mean), self.std(teacher_std)), -1)
        hidden = self.temporal(self.input(x.transpose(1, 2))).transpose(1, 2)
        hidden = hidden*valid[..., None].to(hidden)
        shift = self.max_shift*torch.tanh(self.shift(hidden.transpose(1, 2)).transpose(1, 2))
        shift = shift*valid[..., None].to(shift)
        pooled = hidden.sum(1)/lengths[:, None].to(hidden)
        return {'shift': shift, 'gate_logit': self.gate(pooled).squeeze(-1), 'valid_mask': valid}


def feasible_batch_shift(prediction, desired):
    """Project a shared shift, never independently clip ensemble members."""
    if prediction.ndim != 4 or prediction.shape[1] != 10 or prediction.shape[-1] != 25:
        raise ValueError('prediction must be [B,10,T,25]')
    va = prediction[..., 15:17]
    lower, upper = -1-va.amin(1), 1-va.amax(1)
    return desired.clamp(min=lower, max=upper)


def apply_batch_shift(prediction, shift):
    result = prediction.clone()
    result[..., 15:17] += shift[:, None]
    return result
