"""Versioned initial-state prior, never additive output noise."""
import math
import torch
from .data import draw
from .export import recording_noise
PRIOR_VERSION='shared-plus-local-standardized24-v1'
GLOBAL_SEED=8100123

def make_initial_noise(local_noise,global_noise,rho,valid_mask=None):
    if not 0<=rho<1:raise ValueError('rho outside [0,1)')
    if rho==0:result=local_noise # exact compatibility; no global draw required
    else:
        if global_noise is None or global_noise.shape!=local_noise.shape[:-2]+local_noise.shape[-1:]:raise ValueError('global noise shape')
        if global_noise.device!=local_noise.device or global_noise.dtype!=local_noise.dtype:raise ValueError('global dtype/device mismatch')
        result=math.sqrt(rho)*global_noise.unsqueeze(-2)+math.sqrt(1-rho)*local_noise
    if valid_mask is not None:
        mask=valid_mask
        while mask.ndim<result.ndim-1:mask=mask.unsqueeze(-2)
        result=result.masked_fill(~mask[...,None],0)
    return result

def training_noise(shape,cfg,step,path,rho,mask):
    seed,tag=(cfg.noise_seed,'FM_noise') if path=='FM' else (cfg.rollout_seed,'task_rollout')
    local=draw(shape,seed,step,tag).to(mask.device)
    global_noise=None if rho==0 else draw(shape[:-2]+shape[-1:],GLOBAL_SEED,step,path+'_global').to(mask.device)
    return make_initial_noise(local,global_noise,rho,mask)

def recording_bases(config,source_id,length,k=10):
    local=recording_noise(config,source_id,length,k)
    global_noise=torch.stack([draw((24,),GLOBAL_SEED,f'{source_id}|sample{i}','recording_global') for i in range(k)])
    return local,global_noise

def prior_metadata(rho):
    return dict(version=PRIOR_VERSION,rho=rho,coordinates='TRAIN-standardized 24d reaction coordinates',global_seed=GLOBAL_SEED,global_key='sourceID/sample index, no block or length; independent occurrence/step FM_global and task_global in training',public_noise='base local tensor; optional separate base global tensor; sample combines once; internal rollout accepts composed initial state')
