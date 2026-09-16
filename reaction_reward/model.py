"""Fresh conditional two-head judge; accepts numeric tensors only."""
import torch
from torch import nn
import torch.nn.functional as F
class Stem(nn.Module):
 def __init__(self,d):
  super().__init__();self.local=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Conv1d(256,256,3,padding=1));self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,1024,.1,batch_first=True),2,enable_nested_tensor=False)
 def forward(self,x,t,m):
  h=self.local[1](self.local[0](x)).masked_fill(~m[...,None],0);h=self.local[2](h.transpose(1,2)).transpose(1,2).masked_fill(~m[...,None],0)
  pad=(-h.shape[1])%4;h=F.pad(h,(0,0,0,pad));t=F.pad(t,(0,pad));m=F.pad(m,(0,pad));den=m.reshape(len(m),-1,4).sum(-1).clamp_min(1)
  h=h.reshape(len(h),-1,4,256).sum(2)/den[...,None];t=(t*m).reshape(len(t),-1,4).sum(-1)/den;m=m.reshape(len(m),-1,4).any(-1)
  freq=torch.exp(torch.arange(128,device=h.device)*(-9.21034/128));phase=t[...,None]*freq;pos=torch.cat([phase.sin(),phase.cos()],-1)
  return self.enc(h+pos,src_key_padding_mask=~m),t,m
class Cross(nn.Module):
 def __init__(self):
  super().__init__();self.attn=nn.MultiheadAttention(256,8,.1,batch_first=True);self.bias=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,8));self.norm=nn.LayerNorm(256);self.ff=nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,256));self.norm2=nn.LayerNorm(256)
 def forward(self,q,k,qt,kt,qm,km):
  b=self.bias((qt[:,:,None]-kt[:,None,:])[...,None]).permute(0,3,1,2);b=b.masked_fill(~km[:,None,None,:],float('-inf')).reshape(-1,q.shape[1],k.shape[1]);z=self.attn(q,k,k,attn_mask=b,need_weights=False)[0];q=self.norm(q+z);return self.norm2(q+self.ff(q)).masked_fill(~qm[...,None],0)
class Judge(nn.Module):
 def __init__(self):
  super().__init__();self.x=Stem(851);self.y=Stem(51);self.xy=nn.ModuleList([Cross(),Cross()]);self.yx=nn.ModuleList([Cross(),Cross()]);self.context=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,1));self.temporal=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,1))
 def forward(self,x,y,xt,yt,xm,ym):
  dt=torch.diff(yt,dim=1,prepend=yt[:,:1]);valid=ym & torch.cat([torch.zeros_like(ym[:,:1]),ym[:,:-1]],1);dy=torch.diff(y,dim=1,prepend=y[:,:1])/dt.clamp_min(1e-4)[...,None];dy=dy.masked_fill(~valid[...,None],0)
  a,at,am=self.x(x,xt,xm);b,bt,bm=self.y(torch.cat([y,dy,dt[...,None]],-1),yt,ym)
  for xy,yx in zip(self.xy,self.yx):a,b=xy(a,b,at,bt,am,bm),yx(b,a,bt,at,bm,am)
  pool=lambda z,m:(z*m[...,None]).sum(1)/m.sum(1).clamp_min(1)[...,None]
  ap,bp=pool(a,am),pool(b,bm);c=self.context(torch.cat([ap,bp,ap*bp,(ap-bp).abs()],-1)).squeeze(-1)
  t=pool(self.temporal(torch.cat([b,bp[:,None].expand_as(b)],-1)),bm).squeeze(-1)
  return torch.stack([c,t],-1)
