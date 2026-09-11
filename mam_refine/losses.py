"""Single-variable coverage substitution; legacy task objective remains frozen."""
import itertools
import torch
from mam_target.losses import pair_costs,softmin,ccc25

def one_to_one(cost):
    if cost.ndim!=3 or cost.shape[1:]!=(4,4):raise ValueError('requires K=R=4 slots')
    permutations=torch.tensor(list(itertools.permutations(range(4))),device=cost.device)
    row=torch.arange(4,device=cost.device)[None]
    choice=cost.detach()[:,row,permutations].mean(-1).argmin(-1)
    assignment=permutations[choice]
    return cost.gather(2,assignment[...,None]).squeeze(-1).mean(-1).mean()

def objective(pred,batch,targets,target_lengths,weights,cover_mode):
    c,d=pair_costs(pred,targets,batch['source_lengths'],target_lengths,weights['grid'])
    cost=weights['a']*c+weights['b']*d
    valid=softmin(cost,2).mean()
    if cover_mode=='softmin':cover=softmin(cost,1).mean()
    elif cover_mode=='one_to_one':cover=one_to_one(cost)
    else:raise ValueError('unknown coverage mode')
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
