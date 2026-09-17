"""GT clock representation only. Predicted speaker clock is explicitly not implemented."""
import torch

def clock_features(program,token_count,stride=4):
    n=program.lengths.to(program.timing.dtype);timing=program.timing/n[:,None,None]
    positions=(torch.arange(token_count,device=n.device,dtype=n.dtype)*stride+(stride-1)/2)[None]
    positions=torch.minimum(positions,(n-1)[:,None])/n[:,None]
    onset,peak,offset=timing.unbind(-1)
    q=positions[:,:,None]
    rise=((q-onset[:,None])/(peak-onset).clamp_min(1e-6)[:,None]).clamp(0,1)
    fall=((offset[:,None]-q)/(offset-peak).clamp_min(1e-6)[:,None]).clamp(0,1)
    envelope=torch.minimum(rise,fall)*program.action_mask[:,None]
    return torch.cat([timing,program.intensity[...,None]],-1),envelope
