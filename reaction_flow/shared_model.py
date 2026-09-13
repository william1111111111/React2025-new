"""Same network parameters; persistent correlated initial-prior contract."""
import torch
from .sampler import ReactionFlow,load_checkpoint as old_load
from .shared_noise import make_initial_noise,prior_metadata,PRIOR_VERSION

class SharedFlow(ReactionFlow):
    @torch.no_grad()
    def sample(self,speaker_audio,speaker_emotion,speaker_3dmm,lengths,sample_count=10,noise=None,integration_steps=16,return_aux=False,candidate_chunk=None,position_offset=None,global_noise=None):
        b,t=speaker_audio.shape[:2]
        if noise is None:noise=torch.randn(b,sample_count,t,24,device=speaker_audio.device,dtype=speaker_audio.dtype)
        if global_noise is None and self.rho!=0:global_noise=torch.randn(b,sample_count,24,device=noise.device,dtype=noise.dtype)
        mask=torch.arange(t,device=noise.device)[None]<lengths.to(noise.device)[:,None]
        initial=make_initial_noise(noise,global_noise,self.rho,mask)
        result=super().sample(speaker_audio,speaker_emotion,speaker_3dmm,lengths,sample_count,initial,integration_steps,return_aux,candidate_chunk,position_offset)
        if return_aux:
            # Public noise is always BASE local, including the legacy alias.
            # initial_state is for inspection/internal rollout, not sample(noise=...).
            result.update(noise=noise,local_noise=noise,global_noise=global_noise,
                          initial_state=initial,prior_metadata=prior_metadata(self.rho))
        return result

def upgrade(model,rho):
    # No reinitialization or capacity/state_dict change; subclass only defines prior composition.
    model.__class__=SharedFlow;model.rho=float(rho);return model

def load_checkpoint(path,device='cpu',parent_rho=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format_version')=='reaction-flow-shared24-v1':
        from .config import FlowConfig
        from hirp.config import HiRPConfig
        cfg=dict(saved['flow_config']);cfg['checkpoints']=tuple(cfg['checkpoints'])
        with torch.random.fork_rng(devices=[]):model=SharedFlow(FlowConfig(**cfg),HiRPConfig(**saved['condition_config']))
        model.load_state_dict(saved['model']);model.to(device).eval()
    else:
        if parent_rho is None:raise ValueError('explicit parent migration rho required')
        model,saved=old_load(path,device)
    if 'initial_prior' in saved:
        prior=saved['initial_prior'];assert prior['version']==PRIOR_VERSION
        rho=prior['rho'];assert prior==prior_metadata(rho)
        if parent_rho is not None:assert rho==parent_rho
    elif parent_rho is not None:rho=parent_rho
    else:raise ValueError('missing initial prior metadata; explicit parent migration rho required')
    return upgrade(model,rho),saved

@torch.no_grad()
def export_full(model,*,speaker_audio,speaker_emotion,speaker_3dmm,source_length,noise,global_noise,chunk_T=750,candidate_chunk=10):
    assert noise.shape==(10,source_length,24) and global_noise.shape==(10,24)
    outputs=[]
    for start in range(0,source_length,chunk_T):
        n=min(chunk_T,source_length-start);inputs=[]
        for v in (speaker_audio,speaker_emotion,speaker_3dmm):
            x=v.new_zeros(1,chunk_T,v.shape[-1]);x[0,:n]=v[start:start+n];inputs.append(x)
        z=noise.new_zeros(1,10,chunk_T,24);z[0,:,:n]=noise[:,start:start+n]
        y=model.sample(*inputs,torch.tensor([n],device=z.device),10,z,16,candidate_chunk=candidate_chunk,position_offset=torch.tensor([start],device=z.device),global_noise=global_noise[None])
        outputs.append(y[0,:,:n].cpu())
    return torch.cat(outputs,1)
