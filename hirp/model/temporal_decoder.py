from torch import nn


class FiLM(nn.Module):
    def __init__(self, latent_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(latent_dim, 2*d_model)
        # Exact zero weights block all prior gradients on the first backward.
        nn.init.normal_(self.projection.weight, std=1e-3)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, z):
        gamma, beta = self.projection(z).chunk(2, -1)
        return x * (1 + gamma[:, None]) + beta[:, None]


class TemporalBlock(nn.Module):
    def __init__(self, config, dilation):
        super().__init__()
        d = config.d_model
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.film1, self.film2 = FiLM(config.latent_dim, d), FiLM(config.latent_dim, d)
        self.activation = nn.GELU()
        self.depthwise = nn.Conv1d(d, d, 3, padding=dilation,
                                   dilation=dilation, groups=d)
        self.pointwise = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, config.dim_feedforward), nn.GELU(),
                                 nn.Linear(config.dim_feedforward, d))

    def forward(self, x, z, mask):
        y = self.activation(self.film1(self.norm1(x), z))
        # Mask before convolution too: FiLM/LN biases must not leak from padding.
        y = y.masked_fill(~mask[..., None], 0)
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        x = (x + self.pointwise(y)).masked_fill(~mask[..., None], 0)
        x = x + self.ffn(self.film2(self.norm2(x), z))
        return x.masked_fill(~mask[..., None], 0)


class TemporalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(config, d) for d in config.dilations])

    def forward(self, h, z, valid_mask):
        b, k, _ = z.shape
        t, d = h.shape[1:]
        x = h[:, None].expand(b, k, t, d).reshape(b*k, t, d)
        mask = valid_mask[:, None].expand(b, k, t).reshape(b*k, t)
        z = z.reshape(b*k, -1)
        for block in self.blocks:
            x = block(x, z, mask)
        return x.reshape(b, k, t, d)
