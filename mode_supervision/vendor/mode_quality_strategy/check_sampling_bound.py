"""Synthetic check only. Does not load REACT, a model checkpoint, or BERT."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def mean_pairwise_square(z: np.ndarray) -> np.ndarray:
    if z.ndim != 3 or z.shape[1] < 2:
        raise ValueError('Expected [sets,K,D] with K>=2')
    k = z.shape[1]
    return 2*k/(k-1)*(np.mean(z*z,axis=(1,2))-np.mean(z.mean(axis=1)**2,axis=1))

def main() -> None:
    rng=np.random.default_rng(20260914)
    sets,k,d=40000,10,5
    iid=rng.normal(size=(sets,k,d))
    coupled=(iid-iid.mean(axis=1,keepdims=True))/np.sqrt(1-1/k)
    small=iid[:30]
    direct=np.sum((small[:,:,None,:]-small[:,None,:,:])**2,axis=(1,2,3))/(k*(k-1)*d)
    err=float(np.max(np.abs(direct-mean_pairwise_square(small))))
    assert err < 1e-12
    result=dict(sets=sets,K=k,dimension=d,expected_iid=2.0,
                expected_upper=2*k/(k-1),relative_gain_upper_percent=100/(k-1),
                empirical_iid=float(mean_pairwise_square(iid).mean()),
                empirical_coupled=float(mean_pairwise_square(coupled).mean()),
                pairwise_identity_error=err,
                note='Synthetic Gaussian check; fixed marginal, expectation only.')
    assert abs(result['empirical_iid']-2)<0.02
    assert abs(result['empirical_coupled']-20/9)<0.02
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
