"""Euler16 source-only trajectory sampler. No target-dependent public arguments."""
from dataclasses import asdict
import torch
from torch import nn
from .condition_encoder import ConditionEncoder
from .velocity_model import VelocityModel
from .reaction_transform import ReactionTransform
from .config import FlowConfig
from hirp.config import HiRPConfig

class ReactionFlow(nn.Module):
    def __init__(self,config=None,condition_config=None,mean=None,std=None):
        super().__init__();self.config=config or FlowConfig();self.condition_config=condition_config or HiRPConfig()
        self.condition=ConditionEncoder(self.condition_config);self.velocity=VelocityModel(self.config)
        self.transform=ReactionTransform(mean,std,self.config.eps,self.config.std_floor)
    def train(self,mode=True):
        super().train(mode);self.condition.eval();return self
    def rollout(self,h,source_valid,noise,integration_steps=16,position_offset=None,gradient_checkpointing=False):
        b,k,t,c=noise.shape
        if c!=24 or integration_steps<1 or t!=h.shape[1] or b!=h.shape[0]:raise ValueError('invalid rollout shape or steps')
        hh=h[:,None].expand(-1,k,-1,-1).reshape(b*k,t,-1);mask=source_valid[:,None].expand(-1,k,-1).reshape(b*k,t)
        offset=None if position_offset is None else torch.as_tensor(position_offset,device=noise.device).reshape(b,1).expand(-1,k).reshape(-1)
        u=noise.reshape(b*k,t,24).masked_fill(~mask[...,None],0)
        for step in range(integration_steps):
            tau=u.new_full((b*k,),step/integration_steps)
            u=u+self.velocity(u,tau,hh,mask,mask,offset,gradient_checkpointing)/integration_steps
            u=u.masked_fill(~mask[...,None],0)
        return self.transform.inverse(u).masked_fill(~mask[...,None],0).reshape(b,k,t,25)
    @torch.no_grad()
    def sample(self,speaker_audio,speaker_emotion,speaker_3dmm,lengths,sample_count=10,noise=None,integration_steps=16,return_aux=False,candidate_chunk=None,position_offset=None):
        if not isinstance(sample_count,int) or sample_count<1:raise ValueError('invalid K')
        modes=[(m,m.training) for m in self.modules()]
        try:
            self.eval();h,valid=self.condition(speaker_audio,speaker_emotion,speaker_3dmm,lengths);b,t,_=h.shape
            if noise is None:noise=torch.randn(b,sample_count,t,24,device=h.device,dtype=h.dtype)
            if noise.shape!=(b,sample_count,t,24) or noise.dtype!=h.dtype or noise.device!=h.device:raise ValueError('noise must match [B,K,T,24], dtype and device')
            chunk=candidate_chunk or sample_count
            if chunk<1:raise ValueError('invalid candidate chunk')
            y=torch.cat([self.rollout(h,valid,noise[:,i:i+chunk],integration_steps,position_offset) for i in range(0,sample_count,chunk)],1)
            return dict(predictions=y,noise=noise,NFE=integration_steps,valid_mask=valid) if return_aux else y
        finally:
            for m,mode in modes:m.training=mode

def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format_version')!='reaction-flow-direct24-v1':raise ValueError('not a trajectory flow checkpoint')
    cfg=dict(saved['flow_config']);cfg['checkpoints']=tuple(cfg['checkpoints'])
    with torch.random.fork_rng(devices=[]):model=ReactionFlow(FlowConfig(**cfg),HiRPConfig(**saved['condition_config']))
    model.load_state_dict(saved['model']);model.to(device).eval();return model,saved

def metadata(model):
    return dict(format_version='reaction-flow-direct24-v1',flow_config=asdict(model.config),condition_config=asdict(model.condition_config),solver='Euler',production_NFE=16,output_policy='continuous AU, tanh VA, expression softmax; boundary-only clipping TRAIN transform')
