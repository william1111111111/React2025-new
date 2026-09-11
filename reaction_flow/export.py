"""750-frame source blocks with one full-recording temporal noise tensor."""
import torch
from .data import draw

def recording_noise(config,clip_id,length,k=10):
    # Independent keyed absolute750-frame blocks, assembled once per recording.
    # Sample identity and common-frame prefixes are invariant to K and length.
    candidates=[]
    for sample in range(k):
        blocks=[draw((750,24),config.evaluation_seed,f'{clip_id}|sample{sample}|frame{start}','production_recording_noise') for start in range(0,length,750)]
        candidates.append(torch.cat(blocks,0)[:length])
    return torch.stack(candidates)

@torch.no_grad()
def export_full(model,*,speaker_audio,speaker_emotion,speaker_3dmm,source_length,noise,chunk_T=750,candidate_chunk=10):
    if noise.shape!=(10,source_length,24):raise ValueError('expected independent full recording [10,L,24] noise')
    outputs=[]
    for start in range(0,source_length,chunk_T):
        n=min(chunk_T,source_length-start);inputs=[]
        for x in (speaker_audio,speaker_emotion,speaker_3dmm):
            if x.ndim!=2 or len(x)<source_length:raise ValueError('source too short')
            pad=x.new_zeros(1,chunk_T,x.shape[-1]);pad[0,:n]=x[start:start+n];inputs.append(pad)
        z=noise.new_zeros(1,10,chunk_T,24);z[0,:,:n]=noise[:,start:start+n]
        y=model.sample(*inputs,lengths=torch.tensor([n],device=noise.device),sample_count=10,noise=z,integration_steps=16,candidate_chunk=candidate_chunk,position_offset=torch.tensor([start],device=noise.device))
        outputs.append(y[0,:,:n].cpu())
    result=torch.cat(outputs,1)
    if result.shape!=(10,source_length,25) or not torch.isfinite(result).all():raise RuntimeError('invalid full recording output')
    return result
