"""Independent marginal probes and balanced conditional interaction diagnostics."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .common import *
from .data import Data
from .model import Judge
from .train import ARMS
from .audit import summary

def descriptor(y,t):
 dt=np.maximum(np.diff(t),1e-4);v=np.diff(y,axis=0)/dt[:,None];return np.concatenate([y.mean(0),y.std(0),np.abs(v).mean(0)]).astype(np.float32)
def main():
 torch.set_num_threads(2);data=Data();rs=read(OUT/'REAL_EVIDENCE.json');rs=[r for r in rs if r['family']=='base_context'];fit=[r for r in rs if r['split']=='RM_fit' and r['weight']>0];cal=[r for r in rs if r['split']=='RM_cal' and r['weight']>0];audit=[r for r in rs if r['split']=='RM_audit'];dest=OUT/'probes';dest.mkdir(exist_ok=True)
 def features(records,key):return torch.tensor(np.stack([descriptor(*data.target(r[key],r['n'])) for r in records]),device='cuda')
 assert fit and cal and audit
 fa,fb=features(fit,'A'),features(fit,'B');ca,cb=features(cal,'A'),features(cal,'B');aa,ab=features(audit,'A'),features(audit,'B');torch.manual_seed(123);m=nn.Sequential(nn.Linear(75,128),nn.GELU(),nn.Linear(128,1)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);rng=np.random.default_rng(123);best=None;states={}
 for step in range(4000):
  ids=rng.integers(len(fit),size=32);d=(m(fa[ids])-m(fb[ids])).flatten();loss=F.softplus(-d).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step+1 in [1000,2000,4000]:
   with torch.no_grad():nll=float(F.softplus(-(m(ca)-m(cb))).mean())
   if best is None or nll<best:best=nll;states={k:v.detach().clone() for k,v in m.state_dict().items()};selected=step+1
 m.load_state_dict(states)
 with torch.no_grad():ds=(m(aa)-m(ab)).flatten().cpu().tolist()
 raw=[dict(r,difference=d) for r,d in zip(audit,ds)];write(dest/'Y_only.json',dict(selected_step=selected,cal_NLL=best,audit=summary(raw),descriptor='listener mean,std,mean absolute native velocity; independent 4000-step small MLP',scope='same-X balanced context comparison; listener marginals paired in both label positions'))
 write(dest/'X_only.json',dict(score_difference=0.,pair_accuracy=.5,training='not trained: every comparison has identical X in A/B, so source-only cancels identically',interpretation='algebraic sanity check only, not strong shortcut evidence'))
 interactions={}
 for arm in ARMS:
  sel=read(OUT/'training'/arm/'SELECTION.json');model=Judge().cuda();model.load_state_dict(torch.load(sel['selected_checkpoint'],map_location='cpu',weights_only=False)['model']);model.eval();used=set();rows=[]
  for r in audit:
   key=tuple(sorted([r['source'],r['B']['id']]))
   if key in used:continue
   opposite=next((q for q in audit if q['source']==r['B']['id'] and q['B']['id']==r['source']),None)
   if opposite is None:continue
   used.add(key)
   with torch.no_grad():
    def diff(q):return float((model(*data.batch([q],'A'))-model(*data.batch([q],'B')))[0,0])
    value=diff(r)+diff(opposite)
   rows.append(dict(groups=[r['group'],opposite['group']],I=value,state=r['state'],weight=r['weight']))
  interactions[arm]=dict(pairs=len(rows),mean_I=float(np.mean([r['I'] for r in rows])) if rows else None,rows=rows,additive_unimodal_I=0.,semantic_truth='cross combinations remain statistical weak proxies')
 write(dest/'interaction_2x2.json',interactions)
 print('PROBES COMPLETE',flush=True)
if __name__=='__main__':main()
