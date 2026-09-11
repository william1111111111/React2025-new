"""24 coordinates: logit AU, atanh VA, fixed orthonormal expression contrasts."""
import torch
from torch import nn

def contrast_matrix(dtype=torch.float64):
    q=torch.zeros(8,7,dtype=dtype)
    for j in range(7):
        k=j+1;scale=(k*(k+1))**.5;q[:k,j]=1/scale;q[k,j]=-k/scale
    return q

class ReactionTransform(nn.Module):
    def __init__(self,mean=None,std=None,eps=1e-4,std_floor=1e-3):
        super().__init__();self.eps=eps;self.std_floor=std_floor
        self.register_buffer('Q',contrast_matrix())
        self.register_buffer('mean',torch.zeros(24,dtype=torch.float64) if mean is None else torch.as_tensor(mean,dtype=torch.float64))
        self.register_buffer('std',torch.ones(24,dtype=torch.float64) if std is None else torch.as_tensor(std,dtype=torch.float64).clamp_min(std_floor))
    def regularize(self,y):
        au=y[...,:15].clamp(self.eps,1-self.eps);va=y[...,15:17].clamp(-1+self.eps,1-self.eps)
        p=y[...,17:].clamp_min(self.eps);p=p/p.sum(-1,keepdim=True)
        return torch.cat((au,va,p),-1)
    def coordinates(self,y):
        x=self.regularize(y);p=x[...,17:];q=self.Q.to(y)
        return torch.cat((torch.logit(x[...,:15]),torch.atanh(x[...,15:17]),p.log()@q),-1)
    def forward(self,y):return (self.coordinates(y)-self.mean.to(y))/self.std.to(y)
    def inverse(self,u):
        v=u*self.std.to(u)+self.mean.to(u)
        return torch.cat((v[...,:15].sigmoid(),v[...,15:17].tanh(),(v[...,17:]@self.Q.to(u).T).softmax(-1)),-1)
