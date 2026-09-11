import torch
from mam_target.losses import ccc25,softmin,pair_costs
from hirp.phase21 import conditional_score
GROUPS=(('AU',0,15,1.),('VA',15,17,2.),('expression',17,25,1.))

def r2_terms(pred,batch,weights,c,d):
    cost=weights['a']*c+weights['b']*d
    valid=softmin(cost,2).mean();cover=softmin(cost,1).mean();paired=[]
    for i in range(len(pred)):
        n=int(batch['pair_lengths'][i]);p=pred[i,:,:n];y=batch['paired_target'][i,:n].expand_as(p)
        v=1-ccc25(p,y)+(p-y).abs().mean((-1,-2))
        if n>1:v=v+.1*((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1])).abs().mean((-1,-2))
        paired.append(v.min())
    paired=torch.stack(paired).mean();preserve=conditional_score(pred,batch)['loss']
    return dict(r2=valid+weights['beta']*cover+weights['gamma']*paired+weights['eta']*preserve,valid=valid,cover=cover,paired=paired,preserve=preserve)

def quality_proxies(c,d):return softmin(c,2).mean(1),softmin(d,2).mean(1)

def quality_guard(qc,qd,qc0,qd0,scales):
    gc=torch.relu(qc-qc0.detach()).mean()/scales['s_C'];gd=torch.relu(qd-qd0.detach()).mean()/scales['s_D']
    return gc+gd,gc,gd

def pool8(x,n):
    x=x[:,:n];full=n//8;parts=[]
    if full:parts.append(x[:,:full*8].reshape(x.shape[0],full,8,x.shape[-1]).mean(2))
    if n%8:parts.append(x[:,full*8:].mean(1,keepdim=True))
    return torch.cat(parts,1)

def spread(x):
    k=x.shape[0]
    if k<2:raise ValueError('spread requires >=2 candidates')
    return (x[:,None]-x[None]).square().sum()/(k*(k-1)*x.shape[1]*x.shape[2])

def dispersions(pred,parent,targets,source_lengths,target_lengths,ids):
    current=[];base=[];refs=[];valid=[]
    for i in range(len(pred)):
        seen=set();unique=[]
        for j,key in enumerate(ids[i]):
            if key not in seen:unique.append(j);seen.add(key)
        n=min(int(source_lengths[i]),int(target_lengths[i].min()))
        prow=[];brow=[];rrow=[];mask=[]
        for _,a,b,scale in GROUPS:
            p=pool8(pred[i,...,a:b]/scale,n);q=pool8(parent[i,...,a:b].detach()/scale,n);t=pool8(targets[i,unique,:,a:b].detach()/scale,n)
            for dynamic in (False,True):
                ok=len(unique)>=2 and (not dynamic or p.shape[1]>=2)
                if dynamic:p=p-p.mean(1,keepdim=True);q=q-q.mean(1,keepdim=True);t=t-t.mean(1,keepdim=True)
                prow.append(spread(p) if ok else pred[i].sum()*0);brow.append(spread(q) if ok else q.sum()*0);rrow.append(spread(t) if ok else t.sum()*0);mask.append(ok)
        current.append(torch.stack(prow));base.append(torch.stack(brow));refs.append(torch.stack(rrow));valid.append(mask)
    return torch.stack(current),torch.stack(base).detach(),torch.stack(refs).detach(),torch.tensor(valid,device=pred.device,dtype=torch.bool)

def dispersion_loss(d,d0,ref,valid,epsilon,alpha):
    desired=(d0+alpha*torch.relu(ref-d0)).detach()
    value=(torch.relu(desired-d)/(ref+torch.as_tensor(epsilon,device=d.device,dtype=d.dtype))).square()
    return value[valid].mean() if valid.any() else d.sum()*0

def losses(pred,parent,batch,targets,target_lengths,ids,weights,calibration,u,arm,precomputed_parent=None,alpha_override=None,common_parent_step=False):
    c,d=pair_costs(pred,targets,batch['source_lengths'],target_lengths,weights['grid'])
    with torch.no_grad():
        c0,d0=precomputed_parent if precomputed_parent is not None else pair_costs(parent,targets,batch['source_lengths'],target_lengths,weights['grid'])
        qc0,qd0=quality_proxies(c0,d0)
    qc,qd=quality_proxies(c,d)
    if common_parent_step:
        assert float((pred.detach()-parent).abs().max())<2e-6
        qc0,qd0=qc.detach(),qd.detach() # exact same-parent score; avoid FP32 ReLU sign artifact
    terms=r2_terms(pred,batch,weights,c,d)
    guard,gc,gd=quality_guard(qc,qd,qc0,qd0,calibration);quality=terms['r2']+guard
    dd,dd0,ref,mask=dispersions(pred,parent,targets,batch['source_lengths'],target_lengths,ids)
    ramp=min(u/500.,1.);alpha=.25*ramp if alpha_override is None else alpha_override
    disp=dispersion_loss(dd,dd0,ref,mask,calibration['epsilon'],alpha)
    lam=calibration.get('lambda_max',0.)*ramp if arm!='S0_quality' else 0.
    return dict(loss=quality+lam*disp,quality=quality,guard_C=gc,guard_D=gd,disp=disp,lambda_disp=lam,alpha=alpha,**terms),dict(current=dd,base=dd0,reference=ref,valid=mask)
