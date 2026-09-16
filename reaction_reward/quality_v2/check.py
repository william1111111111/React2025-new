import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from reaction_reward.common import ROOT,read,write
from reaction_reward.data import Data
from .model import Judge
from .train import OUT,pair_difference

def main():
 torch.set_num_threads(2);data=Data();torch.manual_seed(123);a=Judge().cuda();ha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in a.state_dict().values())).hexdigest();torch.manual_seed(123);b=Judge().cuda();hb=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in b.state_dict().values())).hexdigest();assert ha==hb
 qr=[r for r in read(OUT/'QUALITY_RECORDS.json') if __import__('pathlib').Path(r['A']['generated']).exists()][:4];assert len(qr)==4
 b.eval();x=data.batch(qr,'A');before=next(b.parameters()).detach().clone();pred=b(*x);target=torch.tensor([r['target'] for r in qr],device='cuda');loss=F.huber_loss(pred[:,2:],target,reduction='none').mean(0).sum();loss.backward();gc=float(b.quality_C[-1].weight.grad.norm());gd=float(b.quality_D[-1].weight.grad.norm());gx=float(b.x.local[0].weight.grad.norm());assert min(gc,gd,gx)>0
 opt=torch.optim.AdamW(b.parameters(),lr=1e-4);opt.step();delta=float((next(b.parameters()).detach()-before).norm());assert delta>0
 a.quality_C.requires_grad_(False);a.quality_D.requires_grad_(False);a.eval()
 with torch.no_grad():
  v=a(*x);swap=a(*(t.flip(0) for t in x)).flip(0);torch.testing.assert_close(v,swap,atol=2e-6,rtol=2e-5)
 r=[r for r in read(OUT/'RECORDS.json') if r['split']=='RM_fit' and r['family']=='temporal' and r['weight']>0][:2];d=pair_difference(a,data,r,'temporal','A-repaired-RMGen');F.softplus(-d).mean().backward();assert a.quality_C[-1].weight.grad is None and a.quality_D[-1].weight.grad is None
 write(OUT/'SMOKE.json',dict(passed=True,shared_initialization_sha256=ha,quality_C_gradient=gc,quality_D_gradient=gd,source_gradient=gx,parameter_delta=delta,A_quality_head_gradients_absent=True,real_TRAIN_quality_candidates=len(qr),real_active_temporal_pairs=len(r),formal_initialization='fresh seed123; never smoke weights'))
 print('SMOKE PASSED',gc,gd,gx,delta,flush=True)
if __name__=='__main__':main()
