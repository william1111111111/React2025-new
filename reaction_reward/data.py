import functools
import numpy as np
import torch
from .common import *
@functools.lru_cache(maxsize=128)
def load(p):
 x=np.load(p);return x[:,0] if x.ndim==3 and x.shape[1]==1 and x.shape[2]==58 else x
class Data:
 def __init__(self):
  self.rows={r['id']:r for r in read(OUT/'SPLIT_MANIFEST.json')['records']};self.norm=read(OUT/'NORMALIZATION.json')['stats']
 def normalize(self,x,k):
  s=self.norm[k];return (x-np.array(s['mean'],np.float32))/np.array(s['std'],np.float32)
 def source(self,rid,start,n):
  r=self.rows[rid];xs=[self.normalize(load(r['files'][f+'/speaker']['path'])[start:start+n],f+'/speaker') for f in ['audio-features','facial-attributes','coefficients']];t=np.array(r['pts']['speaker'][start:start+n],np.float32);return np.concatenate(xs,-1),t-t[0]
 def target(self,d,n):
  r=self.rows[d['id']];role=d.get('role','listener');s=d.get('start',0)
  if 'generated' in d:y=load(d['generated'])[d['candidate'],:n]
  else:y=load(r['files']['facial-attributes/'+role]['path'])[s:s+n]
  t=np.array(r['pts'][role][s:s+n],np.float32);return self.normalize(y,'facial-attributes/listener'),t-t[0]
 def batch(self,records,which,device='cuda'):
  xs=[];ys=[];xt=[];yt=[];ns=[]
  for r in records:
   n=r['n'];x,t=self.source(r['source'],r['start'],n);y,u=self.target(r[which],n);assert len(x)==len(y)==n;xs.append(x);ys.append(y);xt.append(t);yt.append(u);ns.append(n)
  L=max(ns)
  def pad(a,width=None):return np.stack([np.pad(v,((0,L-len(v)),(0,0)) if v.ndim==2 else (0,L-len(v))) for v in a])
  out=[torch.as_tensor(pad(a),device=device,dtype=torch.float32) for a in [xs,ys,xt,yt]];m=torch.arange(L,device=device)[None]<torch.tensor(ns,device=device)[:,None];return (*out,m,m)
