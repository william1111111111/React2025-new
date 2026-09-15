"""Real TRAIN policy gradient, freeze and resume checks; separate diagnostic budget."""
import json,subprocess,sys
import torch
from .common import OUT,write

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    for label,stop in [('continuous',2),('resumed',1),('resumed',2)]:
        subprocess.run([sys.executable,'-m','reward_policy.train','--arm','R-coverage','--stop',str(stop),'--diagnostic',label],check=True)
    a=torch.load(OUT/'diagnostics/continuous/R-coverage/latest.pt',map_location='cpu',weights_only=True);b=torch.load(OUT/'diagnostics/resumed/R-coverage/latest.pt',map_location='cpu',weights_only=True)
    keys=['policy','optimizer','rng','dual','ema','step'];checks={k:equal(a[k],b[k]) for k in keys};assert all(checks.values()),checks
    assert all(r['generator_frozen'] and r['reward_actions_detached'] and r['gradient_norm']>0 for r in a['rows'])
    assert a['rows'][1]['KL_total']!=0
    write(OUT/'TRAIN_ACCEPTANCE.json',dict(passed=True,continuous_vs_resumed=checks,policy_gradient_nonzero=True,frozen_generator=True,detached_reward_actions=True,actual_diagnostic_rollouts=80,diagnostic_only=True))
    print('real policy acceptance passed',checks,flush=True)
if __name__=='__main__':main()
