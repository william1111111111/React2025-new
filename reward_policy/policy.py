import math
import torch
from torch import nn


def normal_log_prob(z,valid):
    return (-.5*(z.square()+math.log(2*math.pi))*valid).sum(-1)

class Coupling(nn.Module):
    def __init__(self,parity):
        super().__init__();self.register_buffer('fixed',torch.arange(96)%2==parity)
        self.net=nn.Sequential(nn.Linear(96+256,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,192))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def parameters_at(self,x,context,valid):
        fixed=self.fixed&valid;active=~self.fixed&valid
        scale,shift=self.net(torch.cat((x*fixed,context),-1)).chunk(2,-1)
        return .25*scale.tanh()*active,shift*active
    def forward(self,x,context,valid,inverse=False):
        s,t=self.parameters_at(x,context,valid)
        y=(x-t)*(-s).exp() if inverse else x*s.exp()+t
        return y*valid,(-s if inverse else s).double().sum(-1)

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Coupling(i%2) for i in range(4)])
    def transform(self,z0,context,valid):
        x=z0*valid;ld=torch.zeros(x.shape[:-1],device=x.device,dtype=torch.float64)
        for layer in self.layers:
            x,d=layer(x,context,valid);ld=ld+d
        return x,ld
    def inverse(self,z,context,valid):
        x=z*valid;ld=torch.zeros(x.shape[:-1],device=x.device,dtype=torch.float64)
        for layer in reversed(self.layers):
            x,d=layer(x,context,valid,True);ld=ld+d
        return x,ld
    def log_prob(self,z,context,valid):
        x,ld=self.inverse(z,context,valid)
        return normal_log_prob(x.double(),valid)+ld
    def rsample_and_log_prob(self,z0,context,valid):
        z,ld=self.transform(z0,context,valid)
        return z,normal_log_prob(z0.double(),valid)-ld
    @torch.no_grad()
    def sample(self,z0,context,valid):return self.transform(z0,context,valid)[0]


def replace_low_frequency(noise,basis,policy,context):
    """Full recording noise [K,N,24], measured basis [N,r]; no padding."""
    r=basis.shape[1];z0=torch.einsum('nr,knd->krd',basis,noise)
    padded=torch.nn.functional.pad(z0,(0,0,0,4-r)).flatten(1)
    valid=(torch.arange(4,device=noise.device)[:,None].expand(4,24)<r).flatten().expand(len(noise),-1)
    z=policy.sample(padded,context.expand(len(noise),-1),valid)
    delta=(z-padded).reshape(-1,4,24)[:,:r]
    return noise+torch.einsum('nr,krd->knd',basis,delta),z,valid
