import torch
from torch import nn


class OutputHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.base = nn.Linear(d_model, 25)
        self.stochastic = nn.Linear(d_model, 25)

    def forward(self, h, decoded, valid_mask):
        raw = self.base(h)[:, None] + self.stochastic(decoded).tanh()
        reaction = torch.cat((raw[..., :15].sigmoid(), raw[..., 15:17].tanh(),
                              raw[..., 17:].softmax(-1)), -1)
        return reaction.masked_fill(~valid_mask[:, None, :, None], 0)
