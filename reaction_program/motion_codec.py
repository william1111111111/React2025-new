"""25D RVQ codec, stride4; block-specific reconstruction objectives."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from .rvq import ResidualVectorQuantizer

@dataclass(frozen=True)
class CodecConfig:
    stride:int=4
    dim:int=128
    codebook_size:int=512
    levels:int=2
    ccc_weight:float=.1
    velocity_weight:float=.1
    quantization_weight:float=1.

class MotionCodec(nn.Module):
    def __init__(self,config=CodecConfig()):
        super().__init__();self.config=config
        if config.stride!=4:raise ValueError('v0 supports stride4 only')
        d=config.dim
        self.encoder=nn.Sequential(nn.Conv1d(25,d,3,padding=1),nn.GELU(),nn.Conv1d(d,d,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(d,d,4,stride=2,padding=1))
        self.quantizer=ResidualVectorQuantizer(d,config.codebook_size,config.levels)
        self.decoder=nn.Sequential(nn.ConvTranspose1d(d,d,4,stride=2,padding=1),nn.GELU(),nn.ConvTranspose1d(d,d,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(d,25,3,padding=1))
    @staticmethod
    def mask(lengths,t):return torch.arange(t,device=lengths.device)[None]<lengths[:,None]
    def validate(self,y,lengths):
        if y.ndim!=3 or y.shape[-1]!=25 or lengths.shape!=(y.shape[0],) or lengths.dtype not in (torch.int32,torch.int64):raise ValueError('bad codec input')
        if (lengths<1).any() or (lengths>y.shape[1]).any():raise ValueError('bad lengths')
        mask=self.mask(lengths,y.shape[1])
        if not torch.isfinite(y[mask]).all():raise ValueError('nonfinite valid frames')
        return mask
    def encode(self,y,lengths):
        valid=self.validate(y,lengths);t=y.shape[1];yp=y.masked_fill(~valid[...,None],0)
        z=self.encoder(F.pad(yp.transpose(1,2),(0,(-t)%4))).transpose(1,2)
        lm=self.mask((lengths+3)//4,z.shape[1]);z=z.masked_fill(~lm[...,None],0)
        z,ids,loss,usage=self.quantizer(z,lm)
        return dict(latent=z,tokens=ids,quantization_loss=loss,usage=usage,token_mask=lm)
    def decode_latent(self,z,lengths,frames):
        raw=self.decoder(z.transpose(1,2)).transpose(1,2)[:,:frames]
        y=torch.cat([raw[...,:15].sigmoid(),raw[...,15:17].tanh(),raw[...,17:].softmax(-1)],-1)
        mask=self.mask(lengths,frames)
        return y.masked_fill(~mask[...,None],0),raw
    def decode_tokens(self,ids,lengths,frames):return self.decode_latent(self.quantizer.decode(ids),lengths,frames)[0]
    def forward(self,y,lengths):
        out=self.encode(y,lengths);out['reconstruction'],out['logits']=self.decode_latent(out['latent'],lengths,y.shape[1]);return out
    def loss(self,y,lengths,out):
        valid=self.validate(y,lengths);pred=out['reconstruction'];logits=out['logits'];yt=y[valid];pt=pred[valid]
        if ((yt[:,:15]<0)|(yt[:,:15]>1)).any() or (yt[:,17:]<0).any() or not torch.allclose(yt[:,17:].sum(-1),torch.ones_like(yt[:,0]),atol=1e-5):raise ValueError('targets outside native attribute domains')
        au=F.binary_cross_entropy_with_logits(logits[valid][:,:15],yt[:,:15]);va=F.mse_loss(pt[:,15:17],yt[:,15:17])
        expr=F.kl_div(F.log_softmax(logits[valid][:,17:],-1),yt[:,17:],reduction='batchmean')
        # Per-record/per-channel CCC; constant targets supply no correlation supervision.
        ccc=[]
        for b,n in enumerate(lengths.tolist()):
            a=pred[b,:n];z=y[b,:n];ac=a-a.mean(0);zc=z-z.mean(0);vv=zc.square().mean(0);active=vv>1e-8
            if active.any():ccc.append((1-2*(ac*zc).mean(0)/(ac.square().mean(0)+vv+(a.mean(0)-z.mean(0)).square()+1e-8))[active].mean())
        cc=torch.stack(ccc).mean() if ccc else pred.sum()*0
        vm=valid[:,1:]&valid[:,:-1];dv=pred[:,1:]-pred[:,:-1];dy=y[:,1:]-y[:,:-1]
        # Block balancing avoids giving AU 15/25 of the loss by dimension count.
        vel=sum((dv[vm][:,a:b]-dy[vm][:,a:b]).square().mean() for a,b in [(0,15),(15,17),(17,25)])/3 if vm.any() else pred.sum()*0
        q=out['quantization_loss'];c=self.config
        return dict(total=au+va+expr+c.ccc_weight*cc+c.velocity_weight*vel+c.quantization_weight*q,AU=au,VA=va,expression=expr,CCC=cc,velocity=vel,quantization=q)
