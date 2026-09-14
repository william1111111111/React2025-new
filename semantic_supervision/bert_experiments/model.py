"""Shared zero-residual semantic fusion; BERT never lives inside the trainable model."""
import math
import torch
from torch import nn
from semantic_supervision.models.auto_weak import WeakEvent
from .common import ARMS
from .vendor.bert_semantic_experiment.reference_ops import event_time_bias
class Fusion(nn.Module):
 def __init__(self,width=256,heads=8):
  super().__init__();self.width=width;self.heads=heads
  self.content_norm=nn.LayerNorm(768);self.content_projection=nn.Linear(768,width);self.time_projection=nn.Linear(3,width)
  self.query=nn.Linear(width,width);self.key=nn.Linear(width,width);self.value=nn.Linear(width,width);self.output=nn.Linear(width,width);self.null=nn.Parameter(torch.randn(width)*.02)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,h,valid,content,intervals,known,frames,durations,drop_mask,use_bias):
  b,t,d=h.shape;rows=[];bounds=[];times=[];active=[]
  for i,vecs in enumerate(content):
   vs=[self.null];iv=[(0.,0.)];kt=[False];enabled=bool(vecs) and not bool(drop_mask[i]);active.append(enabled)
   if enabled:
    for vec,interval,k in zip(vecs,intervals[i],known[i]):
     v=vec.to(device=h.device,dtype=h.dtype);q=h.new_tensor([interval[0]/durations[i],interval[1]/durations[i],1.] if k else [0.,0.,0.]);vs.append(self.content_projection(self.content_norm(v))+self.time_projection(q));iv.append(interval);kt.append(k)
   rows.append(torch.stack(vs));bounds.append(iv);times.append(kt)
  j=max(len(r) for r in rows);tokens=h.new_zeros(b,j,d);pad=torch.ones(b,j,device=h.device,dtype=torch.bool);iv=h.new_zeros(b,j,2);kt=torch.zeros(b,j,device=h.device,dtype=torch.bool)
  for i,r in enumerate(rows):tokens[i,:len(r)]=r;pad[i,:len(r)]=False;iv[i,:len(r)]=h.new_tensor(bounds[i]);kt[i,:len(r)]=torch.tensor(times[i],device=h.device)
  split=lambda x:x.reshape(b,-1,self.heads,d//self.heads).transpose(1,2)
  q,k,v=map(split,(self.query(h),self.key(tokens),self.value(tokens)))
  logits=(q@k.transpose(-1,-2))/math.sqrt(d//self.heads)
  if use_bias:logits=logits+event_time_bias(frames,iv,kt)[:,None]
  weights=logits.masked_fill(pad[:,None,None,:],-torch.inf).softmax(-1);attn=(weights@v).transpose(1,2).reshape(b,t,d)
  residual=self.output(attn);mask=valid&torch.tensor(active,device=h.device)[:,None];residual=residual.masked_fill(~mask[...,None],0)
  return h+residual
class BertSemanticFlow(nn.Module):
 def __init__(self,base,arm,content_cache=None):
  super().__init__();assert arm in ARMS;self.base=base;self.arm=arm;self.content_cache=content_cache
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(123);self.fusion=Fusion(base.config.d_model,base.config.heads)
  self.byte_embedding=None
  if arm=='P1-byte':
   with torch.random.fork_rng(devices=[]):torch.manual_seed(124);self.byte_embedding=nn.Embedding(257,768,padding_idx=0)
 def semantic_parameters(self):
  yield from self.fusion.parameters()
  if self.byte_embedding is not None:yield from self.byte_embedding.parameters()
 def content(self,text):
  if not text.strip():return None
  if self.arm=='P1-byte':return self.byte_embedding(torch.tensor([c+1 for c in text.encode('utf-8')],device=self.fusion.null.device)).mean(0)
  if self.arm in ('P2-bert','P3-bert-time'):return self.content_cache.get(text)
  return None
 def condition(self,audio,emotion,coefficients,lengths,events,frames,durations,drop_mask):
  if len(events)!=len(audio):raise ValueError('batch mismatch')
  h,valid=self.base.condition(audio,emotion,coefficients,lengths);content=[];intervals=[];known=[]
  for es in events:
   vs=[];iv=[];kt=[]
   for e in es or []:
    if not isinstance(e,WeakEvent) or e.regime!='auto_weak':raise ValueError('only typed legal source events')
    v=self.content(e.text)
    if v is None:continue
    vs.append(v);iv.append(e.interval or (0.,0.));kt.append(e.interval is not None)
   content.append(vs);intervals.append(iv);known.append(kt)
  return self.fusion(h,valid,content,intervals,known,frames.to(h),durations,drop_mask,self.arm=='P3-bert-time'),valid
 @torch.no_grad()
 def sample(self,audio,emotion,coefficients,lengths,events,frames,durations,*,noise,position_offset=None):
  modes=[(m,m.training) for m in self.modules()]
  try:
   self.eval();h,valid=self.condition(audio,emotion,coefficients,lengths,events,frames,durations,[False]*len(events));return self.base.rollout(h,valid,noise,16,position_offset)
  finally:
   for m,mode in modes:m.training=mode
