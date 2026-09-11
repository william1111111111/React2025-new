"""Task loss only on pure-noise Euler samples; FM retains endpoint supervision."""
import argparse,json
import torch
from .train import configure_flow,write,fm_batch
from .config import ROOT,FlowConfig
from .sampler import load_checkpoint
from .data import resources,draw
from hirp.phase15_audit import sha256_file
from mam_target.losses import pair_costs,softmin,ccc25

WEIGHTS=dict(a=1.,b=.15813484622234084,beta=.25,gamma=.25,grid=32)
def task_loss(pred,batch,targets,lengths):
    c,d=pair_costs(pred,targets,batch['source_lengths'],lengths,32);cost=c+WEIGHTS['b']*d
    valid=softmin(cost,2).mean();cover=softmin(cost,1).mean();paired=[]
    for i in range(len(pred)):
        n=int(batch['pair_lengths'][i]);p=pred[i,:,:n];y=batch['paired_target'][i,:n].expand_as(p)
        v=1-ccc25(p,y)+(p-y).abs().mean((-1,-2))
        if n>1:v=v+.1*((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1])).abs().mean((-1,-2))
        paired.append(v.min())
    return valid+.25*cover+.25*torch.stack(paired).mean()

def calibrate(parent):
    cfg=FlowConfig();_,records,data=resources(cfg,'cuda:0');model,meta=load_checkpoint(parent,'cuda:0');assert meta['phase']=='A' and meta['global_step']==12000 and meta['completed'];rows=[];ratios=[]
    for record in records[12000:12016]:
        batch,targets,lengths,ids,y,ns,slots=data.batch(record);fm,h,sm=fm_batch(model,batch,y,ns,record,cfg,'cuda:0')
        params=list(model.velocity.parameters());gg=torch.autograd.grad(fm,params);gn=float(sum(g.square().sum() for g in gg).sqrt());del gg,fm
        noise=draw((4,4,750,24),cfg.rollout_seed,record['step'],'task_rollout').cuda()
        pred=model.rollout(h,sm,noise,16,batch['crop_start'],gradient_checkpointing=True);task=task_loss(pred,batch,targets,lengths)
        gt=torch.autograd.grad(task,params);tn=float(sum(g.square().sum() for g in gt).sqrt())
        if not (gn>0 and tn>1e-12 and torch.isfinite(torch.tensor([gn,tn])).all()):raise RuntimeError('invalid task/FM gradient, do not calibrate')
        ratios.append(.1*gn/tn);rows.append(dict(step=record['step'],FM_gradient=gn,task_gradient=tn));print('CAL',record['step'],gn,tn,flush=True)
    import numpy as np
    write(ROOT/'task_calibration.json',dict(parent_sha256=sha256_file(parent),lambda_task=float(np.median(ratios)),rows=rows,rule='TRAIN16 continuation batches; median .1*FM gradient/task gradient; fixed no DEV',solver='Euler16 differentiable',task_weights=WEIGHTS))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);a=p.parse_args();configure_flow();calibrate(a.parent)
