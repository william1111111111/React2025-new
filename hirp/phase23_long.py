"""Source-only full-recording K10 export; fixed global latent across time chunks."""
import torch
from .phase22 import eval_adapter


def chunks(stream,length,chunk_T):
    if stream.ndim!=2 or length<1 or len(stream)<length or chunk_T<1:raise ValueError('invalid stream/length')
    remainder=length%chunk_T
    count=length//chunk_T+1
    valid=torch.tensor([chunk_T]*(count-1)+[remainder],dtype=torch.long)
    padding=stream.new_zeros(chunk_T-remainder,stream.shape[1])
    return torch.cat([stream[:length],padding]).reshape(count,chunk_T,-1),valid


@torch.no_grad()
def export_full(model,*,speaker_audio,speaker_emotion,speaker_3dmm,source_length,noise,chunk_T=750,candidate_chunk=10):
    if noise.ndim!=2 or noise.shape!=(10,model.config.latent_dim):raise ValueError('global noise must be [10,latent_dim]')
    if candidate_chunk<1:raise ValueError('candidate chunk must be positive')
    streams=[chunks(x,source_length,chunk_T) for x in (speaker_audio,speaker_emotion,speaker_3dmm)]
    if not all(torch.equal(streams[0][1],v[1]) for v in streams):raise ValueError('source chunk lengths disagree')
    outputs=[]
    for i,n in enumerate(streams[0][1].tolist()):
        if n==0:continue  # Official convention's extra zero-valid tail; no valid frames removed.
        inputs=dict(speaker_audio=streams[0][0][i:i+1],speaker_emotion=streams[1][0][i:i+1],speaker_3dmm=streams[2][0][i:i+1],lengths=torch.tensor([n],device=noise.device))
        predictions=[]
        for start in range(0,10,candidate_chunk):
            predictions.append(eval_adapter(model,**inputs,noise=noise[None,start:start+candidate_chunk]))
        outputs.append(torch.cat(predictions,1)[0,:,:n].cpu())
    result=torch.cat(outputs,1)
    if result.shape!=(10,source_length,25):raise RuntimeError('reassembly lost valid source frames')
    return result
