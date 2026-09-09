"""Versioned public prior semantics; historical HiRP/Phase 2.1 stay unchanged."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from . import HiRPConfig, HiRPNet
from .phase21 import OutputConfig, Phase21OutputHead, CHANNEL_SCALE
from .paired_data import paired_model_inputs, pair_valid_mask
from .losses import paired_energy_score
from .train_phase2 import description
from .group_scores import b3_score, b4_score
from .phase15_audit import sha256_file


class HiRP22(HiRPNet):
    def __init__(self, config, output_config, prior_mode):
        super().__init__(config)
        if prior_mode not in ('standard_normal', 'conditional_gaussian'):
            raise ValueError('explicit prior_mode required')
        self.prior_mode = prior_mode
        self.output_config = output_config
        self.output_head = Phase21OutputHead(self.output_head, config.d_model, output_config)
        self.prior.requires_grad_(prior_mode == 'conditional_gaussian')

    def forward(self, speaker_audio, speaker_emotion, speaker_3dmm, lengths,
                sample_count=4, noise=None, return_aux=False):
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError('sample_count must be a positive integer')
        if speaker_audio.ndim != 3:
            raise ValueError('speaker_audio must be [B,T,768]')
        b,t,_=speaker_audio.shape
        for value,width in ((speaker_audio,768),(speaker_emotion,25),(speaker_3dmm,58)):
            if value.shape != (b,t,width):raise ValueError('source shape mismatch')
        if lengths.shape != (b,) or lengths.dtype not in (torch.int32,torch.int64):
            raise ValueError('lengths must be integer [B]')
        if b == 0 or t == 0 or ((lengths < 1) | (lengths > t)).any():
            raise ValueError('each length must be in [1,T]')
        valid=torch.arange(t,device=speaker_audio.device)[None] < lengths.to(speaker_audio.device)[:,None]
        inputs=[v.masked_fill(~valid[...,None],0) for v in (speaker_audio,speaker_emotion,speaker_3dmm)]
        h,context=self.encoder(self.stems(*inputs),valid)
        if self.prior_mode == 'conditional_gaussian':
            mu,log_sigma=self.prior(context)
        else:
            # No conditional prior computation. Match the context precision under AMP.
            mu=context.new_zeros(b,self.config.latent_dim);log_sigma=torch.zeros_like(mu)
        if noise is None:
            noise=torch.randn(b,sample_count,self.config.latent_dim,device=mu.device,dtype=mu.dtype)
        elif noise.shape != (b,sample_count,self.config.latent_dim) or noise.device != mu.device:
            raise ValueError('explicit noise shape/device mismatch')
        elif not noise.is_floating_point():raise ValueError('noise must be floating point')
        noise=noise.to(mu.dtype)
        z=noise if self.prior_mode == 'standard_normal' else mu[:,None]+log_sigma.exp()[:,None]*noise
        predictions=self.output_head(h,self.decoder(h,z,valid),valid)
        if return_aux:
            return dict(predictions=predictions,latent_mu=mu,latent_log_sigma=log_sigma,
                        latent_z=z,noise=noise,context=context,valid_mask=valid)
        return predictions


def make_model(seed, config=None, device='cpu', prior_mode='standard_normal', output_config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return HiRP22(config or HiRPConfig(),output_config or OutputConfig(head_pre_norm=True),prior_mode).to(device)


def model_metadata(model, scales):
    return dict(format_version=22,config=asdict(model.config),output_config=asdict(model.output_config),
                prior_mode=model.prior_mode,latent_dim=model.config.latent_dim,scales=scales)


def load_checkpoint(path, device='cpu', legacy_manifest=None, overrides=None):
    """Only loader. Missing semantic metadata is an error, never an inferred default.

    Historical A0 checkpoints require the matching manifest (or explicit scales).
    Loader construction preserves caller RNG; resume restores checkpoint RNG separately.
    """
    saved=torch.load(path,map_location='cpu',weights_only=True)
    meta={**saved,**(overrides or {})}
    for field in ('config','output_config','prior_mode'):
        if field not in meta:raise ValueError(f'missing checkpoint semantic field: {field}')
    if meta['prior_mode']=='A0':meta['prior_mode']='standard_normal'
    if saved.get('format_version')!=22:
        if 'latent_dim' not in meta['config']:raise ValueError('missing latent_dim')
        meta['latent_dim']=meta['config']['latent_dim']
        if 'scales' not in meta:
            if legacy_manifest is None:raise ValueError('legacy checkpoint requires explicit matching manifest/scales')
            manifest=json.loads(Path(legacy_manifest).read_text())
            if sha256_file(legacy_manifest)!=saved.get('manifest_hash'):raise ValueError('legacy manifest mismatch')
            if 'channel_scale' not in saved or 'scaler_hash' not in manifest:raise ValueError('missing legacy scale metadata')
            meta['scales']=dict(channel_scale=saved['channel_scale'],descriptor_scaler_sha256=manifest['scaler_hash'],
                                descriptor_dimension=75,training_T=manifest['config']['T'])
    for field in ('latent_dim','scales'):
        if field not in meta:raise ValueError(f'missing {field}')
    for field in ('channel_scale','descriptor_scaler_sha256','descriptor_dimension','training_T'):
        if field not in meta['scales']:raise ValueError(f'missing scale metadata: {field}')
    if meta['latent_dim']!=meta['config']['latent_dim']:raise ValueError('latent_dim mismatch')
    scale=torch.tensor(meta['scales']['channel_scale'],dtype=torch.float32)
    if scale.shape!=(25,) or not torch.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('25 finite positive channel scales required')
    if meta['scales']['descriptor_dimension']!=75 or not meta['scales']['descriptor_scaler_sha256']:
        raise ValueError('invalid descriptor metadata')
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        model=HiRP22(HiRPConfig(**meta['config']),OutputConfig(**meta['output_config']),meta['prior_mode'])
    model.load_state_dict(saved['model'],strict=True)
    model.scale_metadata=meta['scales']
    return model.to(device).eval(),saved


def eval_adapter(model, *, speaker_audio, speaker_emotion, speaker_3dmm, lengths, noise):
    return model.sample(speaker_audio,speaker_emotion,speaker_3dmm,lengths,sample_count=noise.shape[1],noise=noise)


def official_adapter(model, *, speaker_audio, speaker_emotion, speaker_3dmm, source_lengths, noise):
    """Ten raw legal-domain reactions; preprocessing/source lengths supplied explicitly.

    No GT selection, reranking or inverse normalization of reaction attributes.
    This source-only adapter is not an official evaluator invocation.
    """
    if noise.shape[1]!=10:raise ValueError('official adapter requires K=10')
    return eval_adapter(model,speaker_audio=speaker_audio,speaker_emotion=speaker_emotion,
                        speaker_3dmm=speaker_3dmm,lengths=source_lengths,noise=noise)


def losses_for_batch(model,batch,noise,scaler,arm,references=None,lambda_group=.1):
    if arm not in ('C0','C1','C2'):raise ValueError('unknown arm')
    if (arm=='C0') != (references is None):raise ValueError('reference contract mismatch')
    pred=model(**paired_model_inputs(batch),sample_count=noise.shape[1],noise=noise)
    conditional=paired_energy_score(pred,batch['paired_target'],pair_valid_mask(batch),
                                     channel_scale=CHANNEL_SCALE,return_details=True)
    group=None
    if arm!='C0':
        phi,valid=description(pred,batch['source_lengths'],scaler)
        group=(b3_score if arm=='C1' else b4_score)(phi,valid,*references)
    return dict(total=conditional['loss']+(lambda_group*group['loss'] if group else 0),
                conditional=conditional,group=group,predictions=pred)
