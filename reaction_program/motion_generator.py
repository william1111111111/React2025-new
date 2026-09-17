"""Minimal masked-token oracle executor. No planner, continuous noise mode or Flow."""
import torch
from torch import nn
import torch.nn.functional as F
from .reaction_clock import clock_features

class OracleProgramExecutor(nn.Module):
    def __init__(self,vocabulary,dim=128,size=512,levels=2):
        super().__init__()
        if not vocabulary or len(set(vocabulary))!=len(vocabulary):raise ValueError('explicit frozen vocabulary required')
        self.vocabulary=tuple(vocabulary);self.size=size;self.levels=levels
        self.action=nn.Embedding(len(vocabulary),dim);self.clock=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        self.token=nn.ModuleList([nn.Embedding(size+1,dim) for _ in range(levels)])
        self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*4,dropout=0.,batch_first=True),2,enable_nested_tensor=False)
        self.readout=nn.Linear(dim,levels*size)
    def forward(self,tokens,program):
        program.validate(len(self.vocabulary));b,l,q=tokens.shape
        if q!=self.levels or b!=len(program.lengths) or ((tokens<0)|(tokens>self.size)).any():raise ValueError('bad motion tokens')
        if l!=int(((program.lengths+3)//4).max()):raise ValueError('tokens must match native stride4 timeline')
        f,envelope=clock_features(program,l);act=self.action(program.action_ids.clamp_min(0))+self.clock(f)
        act=act.masked_fill(~program.action_mask[...,None],0)
        # A global program summary plus explicit local action envelope; supports co-occurrence.
        global_context=act.sum(1)/program.action_mask.sum(1)[:,None]
        local=torch.einsum('bla,bad->bld',envelope,act)/program.action_mask.sum(1)[:,None,None]
        h=sum(emb(tokens[:,:,i]) for i,emb in enumerate(self.token))+local+global_context[:,None]
        pos=torch.arange(l,device=h.device,dtype=h.dtype)[None,:,None]/((program.lengths+3)//4)[:,None,None]
        freq=torch.arange(h.shape[-1],device=h.device,dtype=h.dtype)[None,None,:]+1
        h=h+torch.sin(pos*freq)
        mask=torch.arange(l,device=h.device)[None]<((program.lengths+3)//4)[:,None]
        h=self.body(h,src_key_padding_mask=~mask).masked_fill(~mask[...,None],0)
        return self.readout(h).reshape(b,l,self.levels,self.size),mask
    def loss(self,target_tokens,program,mask_uniform):
        valid=target_tokens>=0
        if mask_uniform.shape!=target_tokens.shape:raise ValueError('mask stream shape mismatch')
        hidden=(mask_uniform<.75)&valid
        # Guarantee some supervision per row without treating padded tokens as targets.
        for b in range(len(hidden)):
            if not hidden[b].any():hidden[b,0,0]=True
        inp=target_tokens.clamp_min(0).masked_fill(hidden,self.size)
        logits,_=self(inp,program)
        return F.cross_entropy(logits[hidden],target_tokens[hidden])
    @torch.no_grad()
    def generate(self,program,uniforms):
        """Explicit [B,K,iterations,L,levels] uniforms preserve same-input/K-prefix."""
        if self.training:raise ValueError('generation requires eval mode')
        b,k,steps,l,q=uniforms.shape
        if b!=len(program.lengths) or q!=self.levels or steps<1 or ((uniforms<0)|(uniforms>=1)).any():raise ValueError('bad sample stream')
        outputs=[]
        for j in range(k):
            tokens=torch.full((b,l,q),self.size,device=uniforms.device,dtype=torch.long)
            for it in range(steps):
                logits,valid=self(tokens,program);cdf=logits.softmax(-1).cumsum(-1)
                draw=(uniforms[:,j,it,...,None]>cdf).sum(-1).clamp_max(self.size-1)
                # Fixed reveal order, not GT/quality based candidate selection.
                positions=torch.arange(l*q,device=tokens.device).reshape(1,l,q)
                reveal=positions<((it+1)*l*q+steps-1)//steps
                tokens=torch.where((tokens==self.size)&reveal,draw,tokens)
            tokens=tokens.masked_fill(~valid[...,None],-1);outputs.append(tokens)
        return torch.stack(outputs,1)
