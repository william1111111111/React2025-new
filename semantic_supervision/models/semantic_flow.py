"""Trainable source semantic path outside frozen ConditionEncoder.no_grad."""
import torch
from torch import nn
from .auto_weak import WeakEvent,TYPES
class SemanticBranch(nn.Module):
    def __init__(self,width,heads,dropout=.1):
        super().__init__();self.byte_embedding=nn.Embedding(257,width,padding_idx=0);self.type_embedding=nn.Embedding(len(TYPES)+1,width);self.time_projection=nn.Linear(3,width);self.projection=nn.Linear(width,width);self.attention=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0.);self.null=nn.Parameter(torch.zeros(width));self.dropout=dropout
    def components(self,event,mode,device,dtype):
        if not isinstance(event,WeakEvent) or event.regime!='auto_weak':raise ValueError('typed auto-weak source input required')
        ids=torch.tensor([x+1 for x in event.text.encode('utf-8')],device=device)
        content=self.byte_embedding(ids).mean(0)
        kind=self.type_embedding(torch.tensor(TYPES.index(event.event_type)+1 if mode=='event' else 0,device=device))
        temporal=torch.tensor([*(event.interval or (0.,0.)),float(event.interval is not None)],device=device,dtype=dtype)
        return content,kind,temporal
    def forward(self,h,events,mode='event'):
        if mode not in ('null','text','event') or len(events)!=len(h):raise ValueError('semantic batch mismatch')
        rows=[]
        for es in events:
            tokens=[self.null]
            if mode!='null' and not (self.training and torch.rand((),device=h.device)<self.dropout):
                for e in es or []:
                    c,k,t=self.components(e,mode,h.device,h.dtype);tokens.append(self.projection(c+k+self.time_projection(t)))
            rows.append(torch.stack(tokens))
        n=max(len(x) for x in rows);tokens=h.new_zeros((len(rows),n,h.shape[-1]));mask=torch.ones((len(rows),n),dtype=torch.bool,device=h.device)
        for i,row in enumerate(rows):tokens[i,:len(row)]=row;mask[i,:len(row)]=False
        return h+self.attention(h,tokens,tokens,key_padding_mask=mask,need_weights=False)[0]
class SemanticFlow(nn.Module):
    def __init__(self,base,dropout=.1):
        super().__init__();self.base=base;self.semantic=SemanticBranch(base.config.d_model,base.config.heads,dropout)
    def condition(self,audio,emotion,coefficients,lengths,events,mode='event'):
        h,valid=self.base.condition(audio,emotion,coefficients,lengths)
        return self.semantic(h,events,mode),valid
    def velocity(self,u,tau,audio,emotion,coefficients,lengths,events,mode='event',position_offset=None):
        h,valid=self.condition(audio,emotion,coefficients,lengths,events,mode)
        return self.base.velocity(u,tau,h,valid,valid,position_offset)
    @torch.no_grad()
    def sample(self,audio,emotion,coefficients,lengths,events,*,noise,mode='event',integration_steps=16,position_offset=None):
        old=self.training
        try:
            self.eval();h,valid=self.condition(audio,emotion,coefficients,lengths,events,mode)
            return self.base.rollout(h,valid,noise,integration_steps,position_offset)
        finally:self.train(old)
