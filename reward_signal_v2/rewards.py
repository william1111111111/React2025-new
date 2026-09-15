import numpy as np

def affinity(phi,target_phi,scales,mask,h2):
    sq=((phi[:,None]-target_phi[None])/scales)**2*mask[None]
    d2=sq.sum(-1)/mask.sum(-1)[None].clip(1)
    return np.exp(-d2/(2*h2)),d2

def coverage_gains(u):
    if len(u)<2:raise ValueError('K>=2 required for this experiment')
    # Per-column differences avoid erasing tiny gains in other target columns.
    return np.array([np.maximum(u[k]-np.delete(u,k,axis=0).max(0),0).mean() for k in range(len(u))])

def quality_credits(q):
    k=len(q)
    if k<2:raise ValueError('independent-candidate baseline needs K>=2')
    others=np.array([np.delete(q,i).mean() for i in range(k)])
    return (q-others)/k
