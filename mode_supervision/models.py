"""Whole-recording conditional flow prior and zero-residual plan injection."""
import torch
from torch import nn

class PlanPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(96, 256)
        self.context = nn.Linear(256, 256)
        self.time = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, 1024, dropout=0., batch_first=True, norm_first=True), 4, enable_nested_tensor=False)
        self.output = nn.Linear(256, 96)

    def forward(self, state, tau, context, positions, mask):
        valid = mask.any(-1)
        if not valid.any(-1).all():
            raise ValueError('empty recording')
        features = torch.cat((positions, tau[:, None, None].expand(-1, state.shape[1], 1)), -1)
        h = self.input(state * mask) + self.context(context) + self.time(features)
        return self.output(self.blocks(h, src_key_padding_mask=~valid)) * mask

    @torch.no_grad()
    def sample(self, noise, context, positions, mask, steps=16):
        state = noise * mask
        for i in range(steps):
            tau = state.new_full((len(state),), i / steps)
            state = (state + self(state, tau, context, positions, mask) / steps) * mask
        return state.detach()

class PlanInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(96, 256), nn.SiLU(), nn.Linear(256, 256))
        self.attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
        self.output = nn.Linear(256, 256)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, h, plan, mask, block_index):
        tokens = self.project(plan * mask)
        current = tokens[torch.arange(len(h), device=h.device), block_index]
        attended, _ = self.attention(current[:, None], tokens, tokens, key_padding_mask=~mask.any(-1), need_weights=False)
        return h + self.output(current + attended[:, 0])[:, None]
