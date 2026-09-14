import types
import numpy as np,pytest,torch
from torch import nn
from .model import Fusion,BertSemanticFlow
from .common import ContentCache,Sources,OUT,read,time_grid
from .vendor.bert_semantic_experiment.reference_ops import masked_content_mean,event_time_bias
from semantic_supervision.models.auto_weak import WeakEvent
class DummyBase(nn.Module):
 def __init__(self):super().__init__();self.config=types.SimpleNamespace(d_model=16,heads=4)
 def condition(self,a,e,c,lengths):return a[...,:16],torch.arange(a.shape[1])[None]<lengths[:,None]
def inputs():
 torch.manual_seed(5);return torch.randn(2,8,768),torch.zeros(2,8,25),torch.zeros(2,8,58),torch.tensor([8,5])
def events():return [[WeakEvent('a','the dog chased the cat','unknown',(0.,.2),(0,5))],None]
def test_pooling_special_pad_empty():
 hidden=torch.tensor([[[99.,99.],[2.,4.],[4.,8.],[999.,999.]]]);mask=torch.tensor([[1,1,1,0]]);special=torch.tensor([[1,0,0,1]])
 assert torch.equal(masked_content_mean(hidden,mask,special),torch.tensor([[3.,6.]]))
 with pytest.raises(ValueError):masked_content_mean(hidden,mask,torch.ones_like(mask))
def test_time_bias_translation_unknown_and_boundaries():
 q=torch.tensor([[0.,1.,3.,6.]]);iv=torch.tensor([[[1.,2.],[float('nan'),float('nan')]]]);known=torch.tensor([[True,False]])
 b=event_time_bias(q,iv,known);assert torch.equal(b,event_time_bias(q+17,iv+17,known));assert torch.equal(b[...,1],torch.zeros_like(b[...,1]));assert b[0,1,0]==b[0,2,0]==0 and b[0,3,0]<0
 pts=np.array([10.,10.1,10.3,10.6]);q,d=time_grid(pts,1,3,3);assert np.allclose(q,[0,.2,.5]) and d==pytest.approx(.7)
def test_all_arms_zero_residual_and_type_ignored():
 cache=ContentCache();text=read(OUT/'legal_texts.json')['train'][0];es=[[WeakEvent('a',text,'unknown',(0,.2),(0,5))],None];a,e,c,l=inputs();states=[]
 for arm in ['P0-null','P1-byte','P2-bert','P3-bert-time']:
  m=BertSemanticFlow(DummyBase(),arm,cache);h,v=m.condition(a,e,c,l,es,torch.arange(8)[None].expand(2,-1)*.04,[.32,.2],[False,False]);assert torch.equal(h,a[...,:16]);states.append(m.fusion.state_dict())
 for s in states[1:]:assert all(torch.equal(s[k],states[0][k]) for k in s)
def test_gradient_ramp_null_padding_and_joint_permutation():
 m=BertSemanticFlow(DummyBase(),'P1-byte');a,e,c,l=inputs();es=events();q=torch.arange(8)[None].expand(2,-1)*.04
 opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for step in range(3):
  opt.zero_grad();h,_=m.condition(a,e,c,l,es,q,[.32,.2],[False,False]);h.square().sum().backward();assert m.fusion.output.weight.grad.norm()>0
  if step==0:assert m.byte_embedding.weight.grad.norm()==0
  else:assert m.byte_embedding.weight.grad.norm()>0
  opt.step()
 h,_=m.condition(a,e,c,l,[None,None],q,[.32,.2],[False,False]);assert torch.equal(h,a[...,:16])
 h,_=m.condition(a,e,c,l,es,q,[.32,.2],[True,False]);assert torch.equal(h,a[...,:16])
 es=[[WeakEvent('a','first','unknown',(0,.1),(0,3)),WeakEvent('b','second','unknown',None,None)],None]
 h,_=m.condition(a,e,c,l,es,q,[.32,.2],[False,False]);hr,_=m.condition(a,e,c,l,[list(reversed(es[0])),None],q,[.32,.2],[False,False]);assert torch.allclose(h,hr,atol=1e-6)
 assert torch.equal(h[1],a[1,:,:16]);assert torch.equal(h[1,5:],a[1,5:,:16])
def test_byte_order_cache_and_source_contract():
 m=BertSemanticFlow(DummyBase(),'P1-byte');assert torch.allclose(m.content('dog cat'),m.content('cat dog'),atol=1e-7)
 cache=ContentCache();text=read(OUT/'legal_texts.json')['train'][0];assert cache.get(text).shape==(768,) and torch.equal(cache.get(text),cache.get(text));assert cache.get('  ') is None
 with pytest.raises(ValueError):cache.get('unregistered arbitrary listener evidence xyz')
 assert len(Sources('train').rows)==48 and len(Sources('val').rows)==80
 m=BertSemanticFlow(DummyBase(),'P0-null');a,e,c,l=inputs()
 with pytest.raises(ValueError):m.condition(a,e,c,l,[[{'text':'listener'}],None],torch.zeros(2,8),[1.,1.],[False,False])
