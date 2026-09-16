"""CPU reference checks, not a REACT training or inference reproduction.

The small NumPy DTW path solver is for testing only. Production should reuse
its pinned, validated native DTW implementation. No corpus/model is loaded.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def old_weights(c, d, c_ref, d_ref, bonus=None):
    c, d = np.asarray(c, float), np.asarray(d, float)
    b = np.zeros_like(c) if bonus is None else np.asarray(bonus, float)
    q = (c-c_ref)/.1 - (d-d_ref)/2.
    raw = q + b
    return .5 + .5*np.clip((raw-raw.mean())/.5, -1., 1.)


def absolute_group_cap(r, c, d, c_ref, d_ref, delta_c=0., delta_d=0.):
    """Illustrative group-level cap; not a calibrated correctness probability."""
    vc = max(0., c_ref-delta_c-float(np.mean(c))) / .1
    vd = max(0., float(np.mean(d))-d_ref-delta_d) / 2.
    if max(vc, vd) > 0.:
        return np.minimum(r, .5-.5*np.tanh(max(vc, vd)))
    return np.asarray(r).copy()


def dtw_path_small(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Expected [T,D] and [S,D]')
    n,m=len(a),len(b)
    if n == 0 or m == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonempty finite sequences required')
    dp=np.full((n+1,m+1),np.inf);dp[0,0]=0.
    parent={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[(i-1,j-1),(i-1,j),(i,j-1)]
            prev=options[int(np.argmin([dp[x,y] for x,y in options]))]
            dp[i,j]=dp[prev]+np.square(a[i-1]-b[j-1]).sum()
            parent[i,j]=prev
    path=[];i,j=n,m
    while i>0 and j>0:
        path.append((i-1,j-1));i,j=parent[i,j]
    return path[::-1],float(np.sqrt(dp[n,m]))


def active_path_distance(x: torch.Tensor, y: torch.Tensor):
    """Exact forward DTW at the current point; selected-path derivative.

    At a nonzero point with a unique optimal path, this is the ordinary local
    gradient. At path ties, it is a chosen branch derivative, not a smooth
    global guarantee. vector_norm has a finite zero subgradient at zero.
    """
    path, score=dtw_path_small(x.detach().cpu().numpy(),y.detach().cpu().numpy())
    ij=torch.tensor(path, device=x.device, dtype=torch.long)
    out=torch.linalg.vector_norm(x.index_select(0,ij[:,0])-y.index_select(0,ij[:,1]))
    assert np.isclose(float(out.detach()), score, atol=1e-10, rtol=1e-10)
    return out, path


def exact_grouped_min(x: torch.Tensor, targets):
    costs=[]
    for y in targets:
        parts=[]
        for lo,hi,w in [(0,15,1/15),(15,17,1.),(17,25,1/8)]:
            a,_=active_path_distance(x[:,lo:hi],y[:,lo:hi])
            parts.append(w*a)
        costs.append(sum(parts))
    index=int(torch.stack(costs).detach().argmin())
    return costs[index],index


def main():
    results={}
    c=np.array([.10,.15,.20,.18]);d=np.array([12.,11.9,11.8,12.2])
    r=old_weights(c,d,.3,10.)
    # The quality ranking remains identical after a common degradation.
    r_bad=old_weights(c-.05,d+1.,.3,10.)
    assert np.allclose(r,r_bad,atol=1e-15)
    results['centered_reward_invariance']={'weights':r.tolist(),'degraded_weights':r_bad.tolist()}
    assert (c<.3).all() and (d>10.).all() and (r>.5).any()
    results['all_worse_can_receive_positive_weight']={'positive_count':int((r>.5).sum())}
    capped=absolute_group_cap(r,c,d,.3,10.)
    assert (capped<.5).all()
    results['absolute_group_failure_no_positive_promotion']={'weights':capped.tolist()}
    # One good metric cannot compensate for violating the other.
    assert (absolute_group_cap(np.array([.8,.9]),[.6,.6],[12.,12.],.3,10.)<.5).all()
    assert np.allclose(absolute_group_cap(np.array([.8,.9]),[.4,.4],[9.,9.],.3,10.),[.8,.9])
    results['two_constraints_separate']=True
    # No target-to-output identity is prescribed; only cost budgets are checked.
    c_t=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    d_t=torch.tensor(12.,dtype=torch.float64,requires_grad=True)
    guard=torch.relu((.3-c_t)/.1).square()+torch.relu((d_t-10.)/2.).square()
    guard.backward()
    assert c_t.grad<0 and d_t.grad>0
    results['guard_gradient_directions']={'ccc_grad':float(c_t.grad),'dtw_grad':float(d_t.grad)}
    # At new==old, neutral core NFT/ref do not prevent real replay from updating.
    w=torch.tensor(1.,dtype=torch.float64,requires_grad=True);old=w.detach().clone()
    nft=.5*(w-2.)**2+.5*(2*old-w-2.)**2
    ref=.01*(w-old)**2
    replay=.25*(w-3.)**2
    gn=float(torch.autograd.grad(nft,w,retain_graph=True)[0])
    total=float(torch.autograd.grad(nft+ref+replay,w)[0])
    assert abs(gn)<1e-14 and abs(total)>0
    results['neutral_NFT_does_not_freeze_FM_replay']={'nft_grad':gn,'total_grad':total}
    rng=np.random.default_rng(27)
    a=rng.normal(size=(6,3));b=rng.normal(size=(5,3))
    x=torch.tensor(a,dtype=torch.float64,requires_grad=True);y=torch.tensor(b,dtype=torch.float64)
    loss,path=active_path_distance(x,y);loss.backward();fd=np.zeros_like(a);h=1e-6
    for idx in np.ndindex(a.shape):
        ap=a.copy();am=a.copy();ap[idx]+=h;am[idx]-=h
        pp,sp=dtw_path_small(ap,b);pm,sm=dtw_path_small(am,b)
        assert pp==path and pm==path
        fd[idx]=(sp-sm)/(2*h)
    err=float(np.abs(fd-x.grad.numpy()).max());assert err<1e-6
    results['active_DTW_gradient_finite_difference']={'max_error':err,'distance':float(loss.detach())}
    z=torch.tensor(a,dtype=torch.float64,requires_grad=True);zero,_=active_path_distance(z,z.detach().clone());zero.backward()
    assert float(zero)==0. and torch.isfinite(z.grad).all() and float(z.grad.abs().max())==0.
    results['zero_DTW_finite_gradient']=True
    xx=torch.tensor(rng.normal(size=(5,25)),dtype=torch.float64,requires_grad=True)
    yy=[torch.tensor(rng.normal(size=(n,25)),dtype=torch.float64) for n in (4,6)]
    val,j=exact_grouped_min(xx,yy);val.backward();assert torch.isfinite(xx.grad).all() and xx.grad.norm()>0
    results['grouped_DTW_nearest_single_target']={'selected_target':j,'distance':float(val.detach()),'grad_norm':float(xx.grad.norm())}
    # Fixed reference budget: don't let successive accepted models lower it.
    fixed=.3;first=.299;second=.298;tol=.0011
    assert first>=fixed-tol and not second>=fixed-tol and second>=first-tol
    results['fixed_reference_prevents_budget_ratcheting']=True
    dest=Path(__file__).parent/'verification_results.json'
    dest.write_text(json.dumps({'scope':'Synthetic CPU checks only; no REACT checkpoint or media loaded','checks_passed':len(results),'checks':results},indent=2))
    print(json.dumps({'checks_passed':len(results),'finite_difference_error':err,'output':str(dest)},indent=2))

if __name__=='__main__':main()
