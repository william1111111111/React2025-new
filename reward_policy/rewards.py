"""Bounded pair-supported coverage and leave-one-out credit, fixed original K."""
import numpy as np


def coverage(u):
    return float(np.max(u,axis=0).mean()) if len(u) else 0.


def utility(ccc,dtw,valid,phi,target_phi,scales,c_floor,d_ceiling):
    delta=(phi[:,None]-target_phi[None])/scales
    similarity=np.exp(-.5*np.mean(delta**2,axis=-1))
    return ((ccc>=c_floor)&(dtw<=d_ceiling)&valid[:,None])*similarity


def distance_bonus(phi,scale,K):
    if len(phi)<2:return 0.
    d=np.mean(((phi[:,None]-phi[None])/scale)**2,axis=-1)
    return float((1-np.exp(-d/2)).sum()/(K*(K-1)))


def difference_advantages(kind,c,d,bad,phi,u,scales,dual):
    K=len(c)
    def score(indices):
        value=coverage(u[indices]) if kind=='R-coverage' else distance_bonus(phi[indices],scales['phi'],K) if kind=='R-distance' else 0.
        # Independent reference constants cancel from all leave-one-out differences.
        return value+dual[0]*c[indices].sum()/K/scales['c']-dual[1]*d[indices].sum()/K/scales['d']-dual[2]*bad[indices].sum()/K
    full=score(np.arange(K))
    return np.array([full-score(np.array([j for j in range(K) if j!=i],dtype=int)) for i in range(K)])
