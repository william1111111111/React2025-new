"""Phase 2.1 opt-in output path and conditional trajectory scoring.

The historical model, Phase 2 trainer, and group estimators remain untouched.
No target/reference is accepted by the generator or forward_a0 adapter.
"""
from dataclasses import dataclass
import torch
from torch import nn
from . import HiRPNet
from .losses import paired_energy_score
from .paired_data import paired_model_inputs, pair_valid_mask
from .phase1_diagnostics import prior_mode
from .train_phase2 import description
from .group_scores import b3_score, b4_score

CHANNEL_SCALE = (1.0,) * 25


@dataclass(frozen=True)
class OutputConfig:
    head_pre_norm: bool = False
    projection_init: str = 'default'
    small_init_std: float = .01


class Phase21OutputHead(nn.Module):
    """Original activation/residual formula, with optional per-frame pre-norm.

    Reuse freshly initialized own-HiRP projections so default old/pre-norm
    comparisons share exactly the same backbone and projection tensors.
    Identity raw_probe exposes the actual differentiable final raw for audit.
    """
    def __init__(self, original, d_model, config):
        super().__init__()
        if config.projection_init not in ('default', 'small'):
            raise ValueError('projection_init must be default or small')
        if config.small_init_std <= 0:
            raise ValueError('small_init_std must be positive')
        self.base, self.stochastic = original.base, original.stochastic
        self.base_norm = nn.LayerNorm(d_model) if config.head_pre_norm else nn.Identity()
        self.stochastic_norm = nn.LayerNorm(d_model) if config.head_pre_norm else nn.Identity()
        self.raw_probe = nn.Identity()
        if config.projection_init == 'small':
            for layer in (self.base, self.stochastic):
                nn.init.normal_(layer.weight, std=config.small_init_std)
                nn.init.zeros_(layer.bias)

    def forward(self, h, decoded, valid_mask):
        raw = self.raw_probe(self.base(self.base_norm(h))[:, None]
                             + self.stochastic(self.stochastic_norm(decoded)).tanh())
        reaction = torch.cat((raw[..., :15].sigmoid(), raw[..., 15:17].tanh(),
                              raw[..., 17:].softmax(-1)), -1)
        return reaction.masked_fill(~valid_mask[:, None, :, None], 0)


def make_model(device='cpu', output_config=None, config=None, seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = HiRPNet(config)
    model.output_head = Phase21OutputHead(model.output_head, model.config.d_model,
                                          output_config or OutputConfig())
    # Frozen deliberately; retained in state_dict and capacity accounting.
    model.prior.requires_grad_(False)
    return model.to(device)


def forward_a0(model, speaker_audio, speaker_emotion, speaker_3dmm, lengths,
               sample_count=4, noise=None, return_aux=False):
    """Source-only forward adapter; cast explicit noise to AMP prior dtype.

    Cast preserves the supplied random numbers to compute precision and their
    autograd connection. Never resamples. Core forward's strict contract stays.
    """
    device_type = speaker_audio.device.type
    # Installed PyTorch 2.1 uses separate CPU/CUDA autocast accessors.
    enabled = torch.is_autocast_cpu_enabled() if device_type == 'cpu' else torch.is_autocast_enabled()
    amp_dtype = torch.get_autocast_cpu_dtype() if device_type == 'cpu' else torch.get_autocast_gpu_dtype()
    dtype = amp_dtype if enabled else next(model.prior.parameters()).dtype
    if noise is not None:
        noise = noise.to(dtype=dtype)  # Device mismatch still rejected by core.
    with prior_mode(model, 'A0'):
        return model(speaker_audio, speaker_emotion, speaker_3dmm, lengths,
                     sample_count=sample_count, noise=noise, return_aux=return_aux)


def conditional_score(predictions, batch, channel_scale=CHANNEL_SCALE):
    return paired_energy_score(predictions, batch['paired_target'], pair_valid_mask(batch),
                               channel_scale=channel_scale, return_details=True)


def losses_for_batch(model, batch, noise, scaler, arm, references=None,
                     lambda_group=0.1, channel_scale=CHANNEL_SCALE):
    if noise.shape[1] < 2:
        raise ValueError('conditional trajectory ES requires K >= 2')
    if arm not in ('C0', 'C1', 'C2'):
        raise ValueError('unknown Phase 2.1 arm')
    if arm == 'C0' and references is not None:
        raise ValueError('C0 must not receive population references')
    if arm != 'C0' and references is None:
        raise ValueError('C1/C2 require session references')
    predictions = forward_a0(model, **paired_model_inputs(batch),
                             sample_count=noise.shape[1], noise=noise)
    conditional = conditional_score(predictions, batch, channel_scale)
    group = None
    if arm != 'C0':
        phi, valid = description(predictions, batch['source_lengths'], scaler)
        group = (b3_score if arm == 'C1' else b4_score)(phi, valid, *references)
    total = conditional['loss'] if group is None else conditional['loss'] + lambda_group * group['loss']
    return dict(total=total, conditional=conditional, group=group, predictions=predictions)
