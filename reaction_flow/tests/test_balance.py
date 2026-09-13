import math
import pytest
from reaction_flow.balance_controller import RatioController,LAMBDA_OLD

def test_controller_measurement_applies_next_step_and_resume():
    c=RatioController(2.,1.);history=[]
    for t in range(1,20):
        history.append(c.weight('B1-ratio',t));c.advance('B1-ratio',t)
    old=c.weight('B1-ratio',20)
    nxt=c.advance('B1-ratio',20,(4.,1.))
    assert c.A==pytest.approx(2.2) and c.B==1.
    assert old==pytest.approx(LAMBDA_OLD+.1*(2-LAMBDA_OLD))
    assert nxt==pytest.approx(LAMBDA_OLD+.105*(2.2-LAMBDA_OLD))
    d=RatioController.restore(c.state('B1-ratio'),'B1-ratio')
    for t in range(21,61):
        assert c.weight('B1-ratio',t)==d.weight('B1-ratio',t)
        args=(3.,2.) if t%20==0 else None
        c.advance('B1-ratio',t,args);d.advance('B1-ratio',t,args)
    assert c.state('B1-ratio')==d.state('B1-ratio')

def test_fixed_weight_caps_and_warmup():
    c=RatioController(100.,1.)
    assert c.cap_hit and c.goal==4*LAMBDA_OLD
    for t in range(1,202):
        assert c.weight('B0-fixed',t)==LAMBDA_OLD
        if t>=200:assert c.weight('B1-ratio',t)==4*LAMBDA_OLD
        c.advance('B0-fixed',t,(100.,1.) if t%20==0 else None)
    assert RatioController(1.,100.).goal==.25*LAMBDA_OLD

@pytest.mark.parametrize('a,b',[(0.,1.),(1.,0.),(math.inf,1.),(1.,math.nan)])
def test_invalid_gradient_norms_fail(a,b):
    with pytest.raises(ValueError):RatioController(a,b)
