import math
import torch
from torch import nn


class SpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        layer = nn.TransformerEncoderLayer(d, config.nhead, config.dim_feedforward,
            config.dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, config.num_layers,
                                              enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.pool = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x, valid_mask):
        t, d = x.shape[1:]
        position = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, d, 2, device=x.device).float()
                         * (-math.log(10000.0) / d))
        pe = torch.zeros(t, d, device=x.device)
        pe[:, 0::2] = torch.sin(position * freq)
        pe[:, 1::2] = torch.cos(position * freq[:d//2])
        h = self.temporal(x + pe.to(x.dtype), src_key_padding_mask=~valid_mask)
        h = h.masked_fill(~valid_mask[..., None], 0)
        mean = h.sum(1) / valid_mask.sum(1, keepdim=True).to(h.dtype)
        scores = (h * self.query).sum(-1) / math.sqrt(d)
        weights = scores.masked_fill(~valid_mask, -torch.inf).softmax(-1)
        pooled = (h * weights[..., None]).sum(1)
        return h, self.pool(torch.cat((mean, pooled), -1))
