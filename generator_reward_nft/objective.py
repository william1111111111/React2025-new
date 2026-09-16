"""Endpoint self-normalized NFT; noise->data time, beta=1. Not log probability."""
import torch

def mean(x,mask):return x.masked_fill(~mask[...,None],0).sum((1,2))/(mask.sum(1)*x.shape[-1])
def core(new,old,target,r,mask):
    old=old.detach();target=target.detach();r=r.detach()
    return (r*mean((new-target).square(),mask)+(1-r)*mean((2*old-new-target).square(),mask)).mean()
def nft(new,old,u,clean,tau,r,mask):
    old=old.detach();clean=clean.detach();u=u.detach();r=r.detach();t=(1-tau[:,None,None])
    plus=u+t*new;minus=u+t*(2*old-new)
    wp=mean((plus-clean).abs(),mask).detach().clamp_min(1e-5)
    wm=mean((minus-clean).abs(),mask).detach().clamp_min(1e-5)
    return (r*mean((plus-clean).square(),mask)/wp+(1-r)*mean((minus-clean).square(),mask)/wm).mean()
