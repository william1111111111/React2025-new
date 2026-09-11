"""FP32 task surrogates, NOT exact FRD. Independent K candidates, target loss only."""
import math
import torch
import torch.nn.functional as F

def ccc25(pred,target):
    """Broadcast [...,T,25]; equivalent to official population CCC incl constants."""
    p=pred-pred.mean(-2,keepdim=True);q=target-target.mean(-2,keepdim=True)
    return (2*(p*q).mean(-2)/(p.square().mean(-2)+q.square().mean(-2)+(pred.mean(-2)-target.mean(-2)).square()+1e-8)).mean(-1)

def soft_dtw(x,y,gamma=.1):
    # Anti-diagonal vectorization: O(N+M) Python launches, not N*M cell loops.
    cost=(x[:,:,None]-y[:,None]).square().sum(-1)
    b,n,m=cost.shape;r=cost.new_full((b,n+1,m+1),float('inf'));r[:,0,0]=0
    for d in range(2,n+m+1):
        i=torch.arange(max(1,d-m),min(n,d-1)+1,device=x.device);j=d-i
        prev=torch.stack((r[:,i-1,j],r[:,i,j-1],r[:,i-1,j-1]),-1)
        r[:,i,j]=cost[:,i-1,j-1]-gamma*torch.logsumexp(-prev/gamma,-1)
    return r[:,n,m]

def grid(x,n=32):
    position=torch.linspace(0,x.shape[-2]-1,n,device=x.device,dtype=x.dtype)
    low=position.floor().long();high=position.ceil().long();w=(position-low)[None,:,None]
    return x.index_select(-2,low)*(1-w)+x.index_select(-2,high)*w

def divergence(x,y,gamma=.1):
    return soft_dtw(x,y,gamma)-.5*(soft_dtw(x,x,gamma)+soft_dtw(y,y,gamma))

def pair_costs(pred,targets,lengths,target_lengths,grid_size=32):
    # Each target's paired/common mask is applied BEFORE grid resampling.
    b,k,t,c=pred.shape;r=targets.shape[1];cs=[];xs=[];ys=[]
    for bi in range(b):
        cp=[]
        for j in range(r):
            n=min(int(lengths[bi]),int(target_lengths[bi,j]))
            if n<2:raise ValueError('task CCC needs at least two valid frames')
            x=pred[bi,:,:n].float();y=targets[bi,j,:n].float().expand(k,-1,-1)
            cp.append(1-ccc25(x,y));xs.append(grid(x,grid_size));ys.append(grid(y,grid_size))
        cs.append(torch.stack(cp,-1))
    xx=torch.cat(xs);yy=torch.cat(ys);terms=[]
    for s,e,w in ((0,15,1/15),(15,17,1),(17,25,1/8)):
        terms.append(w*divergence(xx[...,s:e],yy[...,s:e])/grid_size)
    d=sum(terms).reshape(b,r,k).transpose(1,2)
    return torch.stack(cs),d

def softmin(x,dim,tau=.1):
    return -tau*(torch.logsumexp(-x/tau,dim)-math.log(x.shape[dim]))

def objective(pred,batch,targets,target_lengths,weights):
    c,d=pair_costs(pred,targets,batch['source_lengths'],target_lengths,weights['grid'])
    cost=weights['a']*c+weights['b']*d
    valid=softmin(cost,2).mean();cover=softmin(cost,1).mean()
    paired=[]
    for i in range(len(pred)):
        n=int(batch['pair_lengths'][i]);p=pred[i,:,:n];y=batch['paired_target'][i,:n].expand_as(p)
        v=1-ccc25(p,y)+(p-y).abs().mean((-1,-2))
        if n>1:v=v+.1*((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1])).abs().mean((-1,-2))
        paired.append(v.min())
    paired=torch.stack(paired).mean()
    from hirp.phase21 import conditional_score
    preserve=conditional_score(pred,batch)['loss']
    loss=valid+weights['beta']*cover+weights['gamma']*paired+weights['eta']*preserve
    return dict(loss=loss,valid=valid,cover=cover,paired=paired,preserve=preserve,ccc_cost=c.mean(),sdtw_cost=d.mean())
