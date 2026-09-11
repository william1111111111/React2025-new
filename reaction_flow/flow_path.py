import torch

def flow_path(target,noise,tau,valid):
    t=tau[:,None,None];u=((1-t)*noise+t*target).masked_fill(~valid[...,None],0)
    velocity=(target-noise).masked_fill(~valid[...,None],0)
    return u,velocity

def flow_loss(predicted,target,valid):
    if not valid.any(1).all():raise ValueError('empty selected target')
    value=(predicted-target).float().square().masked_fill(~valid[...,None],0)
    return (value.sum((1,2))/(valid.sum(1)*predicted.shape[-1])).mean()
