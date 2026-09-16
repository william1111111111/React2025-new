import copy
import torch
import torch.nn.functional as F
from reward_policy.environment import TrainEnvironment
from .common import digest

class Models:
    def __init__(self):
        self.env=TrainEnvironment();self.ref=self.env.generator.model.base.velocity
        self.old=copy.deepcopy(self.ref).eval().requires_grad_(False)
        self.student=copy.deepcopy(self.ref).eval().requires_grad_(True)
        assert len(self.student.blocks)==8
        self.ref_hash=digest(self.ref)
    @torch.no_grad()
    def generate(self,e,noise,velocity):
        clean=[];pred=[];mask=e['mask'];h=e['h'];start=e['start'];n=e['n']
        for k in range(0,len(noise),2):
            u=F.pad(noise[k:k+2,start:start+n],(0,0,0,750-n));m=mask.expand(len(u),-1);hh=h.expand(len(u),-1,-1);offset=torch.full((len(u),),start,device=u.device)
            u=u.masked_fill(~m[...,None],0)
            for step in range(16):u=(u+velocity(u,u.new_full((len(u),),step/16),hh,m,m,offset)/16).masked_fill(~m[...,None],0)
            clean.append(u.cpu());pred.append(self.env.generator.model.base.transform.inverse(u)[:,:n].float().cpu())
        return torch.cat(clean),torch.cat(pred)
    @torch.no_grad()
    def synchronize(self):
        for o,s in zip(self.old.parameters(),self.student.parameters()):o.lerp_(s,.5)
