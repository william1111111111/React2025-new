"""Frozen P2; public generation accepts speaker context and noise, never targets."""
import torch
import torch.nn.functional as F
from reaction_flow.sampler import ReactionFlow
from reaction_flow.config import FlowConfig
from hirp.config import HiRPConfig
from semantic_supervision.bert_experiments.model import BertSemanticFlow
from semantic_supervision.bert_experiments.common import ContentCache,time_grid
from mode_supervision.plans import basis
from .common import PARENT,read,OUT,sha
from .policy import replace_low_frequency

class Generator:
    def __init__(self):
        s=torch.load(PARENT,map_location='cpu',weights_only=True);assert sha(PARENT)==read(OUT/'PROTOCOL.json')['parent_sha256']
        cfg=s['base_metadata']['flow_config'];cfg['checkpoints']=tuple(cfg['checkpoints']);base=ReactionFlow(FlowConfig(**cfg),HiRPConfig(**s['base_metadata']['condition_config']))
        self.model=BertSemanticFlow(base,'P2-bert',ContentCache());self.model.load_state_dict(s['model']);self.model.cuda().double().eval().requires_grad_(False)
    @torch.no_grad()
    def features(self,streams,pts,events_by_block):
        hs=[];masks=[]
        for bi,start in enumerate(range(0,len(pts),750)):
            n=min(750,len(pts)-start);q,d=time_grid(pts,start,n)
            xs=[F.pad(x[start:start+n],(0,0,0,750-n))[None].cuda().double() for x in streams]
            h,m=self.model.condition(*xs,torch.tensor([n],device='cuda'),[events_by_block[bi]],torch.tensor(q[None],device='cuda'),[d],[False]);hs.append(h);masks.append(m)
        context=sum((h*m[...,None]).sum(1) for h,m in zip(hs,masks))/len(pts)
        return hs,masks,context
    @torch.no_grad()
    def block(self,h,mask,noise,start,n):
        outputs=[]
        for k in range(0,len(noise),2):
            z=F.pad(noise[k:k+2,start:start+n],(0,0,0,750-n))[None]
            outputs.append(self.model.base.rollout(h,mask,z,16,torch.tensor([start],device='cuda'))[0,:,:n])
        return torch.cat(outputs,0)
    @torch.no_grad()
    def sample(self,streams,pts,events_by_block,noise,policy):
        hs,masks,c=self.features(streams,pts,events_by_block);q=basis(pts).to(noise)
        adjusted,_,_=replace_low_frequency(noise,q,policy,c)
        return torch.cat([self.block(h,m,adjusted,start,min(750,len(pts)-start)) for h,m,start in zip(hs,masks,range(0,len(pts),750))],1)
