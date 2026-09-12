"""Fixed-time diagnostics and target-difference reconstruction, no inference changes."""
import torch
GROUPS=(('AU',0,15),('VA',15,17),('expression',17,25))
def p8(y):
    return torch.cat([y[...,i:i+8,:].mean(-2,keepdim=True).expand_as(y[...,i:i+8,:]) for i in range(0,y.shape[-2],8)],-2)
def spread(y):
    k=y.shape[0]
    if k<2:raise ValueError('requires >=2 candidates')
    return 2*y.var(0,unbiased=True).mean()
def decomposition(y):
    dc=y.mean(-2,keepdim=True).expand_as(y);pool=p8(y)
    parts={'DC':dc,'slow':pool-dc,'fast':y-pool}
    total=spread(y);values={k:spread(v) for k,v in parts.items()}
    return total,values

def dynamic_loss(pred,target,lengths,tau,scales):
    terms=[]
    for i in range(len(pred)):
        if not .5<=float(tau[i])<=.95:continue
        n=int(lengths[i])
        for gi,(_,a,b) in enumerate(GROUPS):
            for si,s in enumerate((1,4,16)):
                if n<=s:continue
                dp=(pred[i,s:n,a:b]-pred[i,:n-s,a:b])/s
                dy=(target[i,s:n,a:b]-target[i,:n-s,a:b])/s
                terms.append((dp-dy).abs().mean()/scales[gi,si])
    return torch.stack(terms).mean() if terms else pred.sum()*0
