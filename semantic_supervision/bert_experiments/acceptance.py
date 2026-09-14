"""Actual T0 step-zero/K-prefix and real three-step optimizer-resume checks."""
import gc,json
from dataclasses import replace
import numpy as np,torch
from .common import *
from .train import build,batch_inputs
from reaction_flow.data import FlowData,draw
from reaction_flow.train import configure_flow
from .model import BertSemanticFlow

def equal_tree(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b
@torch.no_grad()
def main():
 configure_flow();torch.set_num_threads(4)
 c=torch.load(OUT/'diagnostics/continuous3/checkpoints/step_000003.pt',map_location='cpu',weights_only=True);r=torch.load(OUT/'diagnostics/resumed3/checkpoints/step_000003.pt',map_location='cpu',weights_only=True)
 assert equal_tree(c['model'],r['model']) and equal_tree(c['optimizer'],r['optimizer']) and equal_tree(c['rng'],r['rng'])
 assert json.dumps([x['exposure'] for x in c['rows']],sort_keys=True)==json.dumps([x['exposure'] for x in r['rows']],sort_keys=True)
 assert c['rows'][0]['semantic_grads']['fusion.output.weight']>0 and c['rows'][0]['semantic_grads']['fusion.content_projection.weight']==0 and c['rows'][2]['semantic_grads']['fusion.content_projection.weight']>0
 model,opt,parent=build('P0-null');sources=Sources('train');data=FlowData(model.base.config,'cuda:0');rec=read(OUT/'schedule.json')[0];b,y,n,ids,slots,es,q,ds=batch_inputs(data,sources,rec);i=next(i for i,e in enumerate(es) if e);inputs=[b[k][i:i+1].double() for k in ['speaker_audio','speaker_emotion','speaker_3dmm']];lengths=b['source_lengths'][i:i+1];offset=b['crop_start'][i:i+1];frames=q[i:i+1].double();events=[es[i]];durations=[ds[i]]
 noise=draw((1,2,750,24),123,0,'acceptance_noise').cuda().double();base=model.base.double().eval();reference=base.sample(*inputs,lengths,sample_count=2,noise=noise,position_offset=offset);results=[]
 # Reuse identical base parameters while independently constructing each fusion.
 for arm in ARMS:
  m=BertSemanticFlow(base,arm,ContentCache()).cuda().double().eval();out=m.sample(*inputs,lengths,events,frames,durations,noise=noise,position_offset=offset);err=float((out-reference).abs().max());assert err<1e-10
  prefix=m.sample(*inputs,lengths,events,frames,durations,noise=noise[:,:1],position_offset=offset);repeat=m.sample(*inputs,lengths,events,frames,durations,noise=noise,position_offset=offset)
  kerr=float((out[:,:1]-prefix).abs().max());assert kerr<1e-10 and torch.equal(out,repeat);results.append(dict(arm=arm,step0_parent_max_abs=err,K_prefix_max_abs=kerr,repeat_max_abs=0.,fusion_hash=state_hash(m.fusion.state_dict())))
  if arm=='P3-bert-time':
   m.load_state_dict(c['model']);m.double().eval();correct=m.sample(*inputs,lengths,events,frames,durations,noise=noise,position_offset=offset);text=read(OUT/'legal_texts.json')['train'];alternative=next(t for t in text if t!=events[0][0].text);wrong=[list(events[0])];wrong[0][0]=replace(wrong[0][0],text=alternative)
   changed=m.sample(*inputs,lengths,wrong,frames,durations,noise=noise,position_offset=offset);results[-1]['after3_content_prediction_max_change']=float((changed-correct).abs().max())
   h,valid=m.condition(*inputs,lengths,[None],frames,durations,[False]);h0,_=base.condition(*inputs,lengths);assert torch.equal(h,h0)
  del m
 assert len({r['fusion_hash'] for r in results})==1
 write(OUT/'ACCEPTANCE.json',dict(passed=True,real_batch=dict(B=1,T=750,K=2,non_null_events=len(events[0])),real_training=dict(continuous_updates=3,resumed_updates=3,model_exact_match=True,optimizer_exact_match=True,RNG_exact_match=True,first_output_grad=c['rows'][0]['semantic_grads']['fusion.output.weight'],third_upstream_grad=c['rows'][2]['semantic_grads']['fusion.content_projection.weight']),arms=results));print('ACCEPTANCE PASSED',flush=True)
if __name__=='__main__':main()
