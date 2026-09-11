import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

def sine_position(position,width):
    f=torch.exp(torch.arange(0,width,2,device=position.device,dtype=torch.float32)*(-math.log(10000.)/width))
    out=position[...,None]*f;return torch.stack((out.sin(),out.cos()),-1).flatten(-2)

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.d_model
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.n3=nn.LayerNorm(d)
        self.self_attention=nn.MultiheadAttention(d,c.heads,dropout=0.,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(d,c.heads,dropout=0.,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,c.ff_width),nn.GELU(),nn.Linear(c.ff_width,d))
        self.time_shift=nn.Linear(d,d)
    def forward(self,x,h,time,valid,source_valid):
        a=self.n1(x)+self.time_shift(time)[:,None]
        x=x+self.self_attention(a,a,a,key_padding_mask=~valid,need_weights=False)[0]
        a=self.n2(x);x=x+self.cross_attention(a,h,h,key_padding_mask=~source_valid,need_weights=False)[0]
        x=x+self.ff(self.n3(x));return x.masked_fill(~valid[...,None],0)

class VelocityModel(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;self.input=nn.Linear(24,c.d_model)
        self.time=nn.Sequential(nn.Linear(c.d_model,c.d_model),nn.SiLU(),nn.Linear(c.d_model,c.d_model))
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.blocks)]);self.norm=nn.LayerNorm(c.d_model);self.output=nn.Linear(c.d_model,24)
        nn.init.normal_(self.output.weight,std=c.projection_std);nn.init.zeros_(self.output.bias)
    def forward(self,u,tau,h,valid,source_valid,position_offset=None,gradient_checkpointing=False):
        b,t,_=u.shape
        if tau.shape!=(b,) or valid.shape!=(b,t) or not valid.any(1).all() or not source_valid.any(1).all():raise ValueError('invalid state masks/time')
        position=torch.arange(t,device=u.device).float()[None].expand(b,-1)
        if position_offset is not None:position=position+torch.as_tensor(position_offset,device=u.device).reshape(-1,1)
        x=self.input(u.masked_fill(~valid[...,None],0))+sine_position(position,self.config.d_model).to(u)
        time=self.time(sine_position(tau.float()*1000,self.config.d_model).to(u))
        for block in self.blocks:
            if gradient_checkpointing and torch.is_grad_enabled():x=checkpoint(block,x,h,time,valid,source_valid,use_reentrant=False)
            else:x=block(x,h,time,valid,source_valid)
        return self.output(self.norm(x)).masked_fill(~valid[...,None],0)
