from torch import nn


class ConditionalPrior(nn.Module):
    def __init__(self, d_model, latent_dim):
        super().__init__()
        self.mu = nn.Linear(d_model, latent_dim)
        self.log_sigma = nn.Linear(d_model, latent_dim)

    def forward(self, context):
        return self.mu(context), self.log_sigma(context).clamp(-4, 2)
