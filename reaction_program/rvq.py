"""Masked residual vector quantization; no pretrained image models."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    def __init__(self,dim=128,size=512,levels=2,commitment=.25):
        super().__init__();self.size=size;self.levels=levels;self.commitment=commitment
        self.books=nn.Parameter(torch.randn(levels,size,dim)*.1)
    def forward(self,z,valid):
        if z.ndim!=3 or valid.shape!=z.shape[:2] or not valid.any():raise ValueError('invalid latent mask')
        residual=z;total=torch.zeros_like(z);loss=z.sum()*0;ids=[];usage=[]
        for book in self.books:
            distances=residual.square().sum(-1,keepdim=True)+book.square().sum(-1)-2*residual@book.T
            index=distances.argmin(-1);q=F.embedding(index,book).masked_fill(~valid[...,None],0)
            loss=loss+(q[valid]-residual.detach()[valid]).square().mean()+self.commitment*(residual[valid]-q.detach()[valid]).square().mean()
            total=total+q;residual=residual-q.detach();ids.append(index.masked_fill(~valid,-1))
            usage.append(int(index[valid].unique().numel()))
        straight=z+(total-z).detach()
        return straight.masked_fill(~valid[...,None],0),torch.stack(ids,-1),loss/self.levels,usage
    def decode(self,ids):
        if ids.shape[-1]!=self.levels or ((ids<-1)|(ids>=self.size)).any():raise ValueError('bad RVQ ids')
        return sum(F.embedding(ids[...,i].clamp_min(0),book)*(ids[...,i]>=0)[...,None] for i,book in enumerate(self.books))
