import itertools
import numpy as np
from .rewards import quality_credits,coverage_gains,affinity

def test_quality_unbiased():
    p=.3;exact=estimate=0.
    for seq in itertools.product((0,1),repeat=3):
        z=np.array(seq,dtype=float);q=4+2*z;prob=np.prod(np.where(z,p,1-p));score=z-p
        exact+=prob*q.mean()*score.sum();estimate+=prob*np.dot(quality_credits(q),score)
    assert abs(exact-estimate)<1e-12 and abs(exact)>1e-3
    assert np.allclose(quality_credits(np.array([1.,3.,5.])),quality_credits(np.array([1.,3.,5.])+100))

def test_separate_tiny_gains():
    u=np.array([[1.,0.],[1.,1e-200],[0.,0.]])
    g=coverage_gains(u);assert g[1]==5e-201 and g[0]==0 and g[2]==0
    assert np.allclose(coverage_gains(np.array([[1.,0.],[0.,1.]])),[.5,.5])

def test_mask_bandwidth():
    p=np.array([[1.,10000.]]);t=np.zeros((1,2));m=np.array([[True,False]])
    s,d=affinity(p,t,np.ones(2),m,1/(2*np.log(2)))
    assert d[0,0]==1 and abs(s[0,0]-.5)<1e-12
