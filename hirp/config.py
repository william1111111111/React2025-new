from dataclasses import dataclass


@dataclass(frozen=True)
class HiRPConfig:
    d_model: int = 256
    latent_dim: int = 32
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
