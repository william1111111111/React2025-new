import torch
from torch import nn
from ..config import HiRPConfig
from .modality_stem import ModalityStem
from .speaker_encoder import SpeakerEncoder
from .conditional_prior import ConditionalPrior
from .temporal_decoder import TemporalDecoder
from .output_head import OutputHead


class HiRPNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or HiRPConfig()
        c = self.config
        self.stems = ModalityStem(c.d_model)
        self.encoder = SpeakerEncoder(c)
        self.prior = ConditionalPrior(c.d_model, c.latent_dim)
        self.decoder = TemporalDecoder(c)
        self.output_head = OutputHead(c.d_model)

    def forward(self, speaker_audio, speaker_emotion, speaker_3dmm, lengths,
                sample_count=4, noise=None, return_aux=False):
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError('sample_count must be a positive integer')
        if speaker_audio.ndim != 3:
            raise ValueError('speaker_audio must have shape [B,T,768]')
        b, t, _ = speaker_audio.shape
        for value, width in ((speaker_audio, 768), (speaker_emotion, 25), (speaker_3dmm, 58)):
            if value.shape != (b, t, width):
                raise ValueError(f'expected input shape {(b, t, width)}')
        if lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('lengths must be integer [B]')
        if b == 0 or t == 0 or ((lengths < 1) | (lengths > t)).any():
            raise ValueError('each length must be in [1,T]; empty sequences are unsupported')
        valid = torch.arange(t, device=speaker_audio.device)[None] < lengths.to(speaker_audio.device)[:, None]
        inputs = [v.masked_fill(~valid[..., None], 0)
                  for v in (speaker_audio, speaker_emotion, speaker_3dmm)]
        h, context = self.encoder(self.stems(*inputs), valid)
        mu, log_sigma = self.prior(context)
        if noise is None:
            noise = torch.randn(b, sample_count, self.config.latent_dim, device=mu.device, dtype=mu.dtype)
        elif noise.shape != (b, sample_count, self.config.latent_dim):
            raise ValueError('noise must have shape [B,sample_count,latent_dim]')
        elif noise.device != mu.device or noise.dtype != mu.dtype:
            raise ValueError('noise device and dtype must match prior output')
        z = mu[:, None] + log_sigma.exp()[:, None] * noise
        predictions = self.output_head(h, self.decoder(h, z, valid), valid)
        if return_aux:
            return dict(predictions=predictions, latent_mu=mu, latent_log_sigma=log_sigma,
                        latent_z=z, noise=noise, context=context, valid_mask=valid)
        return predictions

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """Temporarily disable dropout, preserving every module's training mode."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self(*args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training
