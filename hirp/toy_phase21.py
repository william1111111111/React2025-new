"""Conditional Bernoulli recovery from non-optimal initializations.

Closed-form expected ES and a tiny logistic network fit to sampled paired labels.
This evaluates loss connections, not REACT generalization or HiRP expressivity.
"""
import json
import torch
from .phase15_audit import canonical_hash


def bernoulli_es(p,q):
    # Exact Euclidean ES on {-1,+1}: 2[p+q-2pq] - 2p(1-p).
    return 2*(p+q-2*p*q)-2*p*(1-p)


def objective(p,q,arm,weight):
    cond = bernoulli_es(p,q).mean()
    if arm == 'C1': return cond+weight*bernoulli_es(p,p.new_tensor(.5)).mean()
    if arm == 'C2': return cond+weight*bernoulli_es(p.mean(),p.new_tensor(.5))
    return cond


def run_toy(weight=.1, steps=500, sample_count=8192):
    q = torch.tensor([.1,.9],dtype=torch.float64)
    rng = torch.Generator().manual_seed(2103)
    labels = (torch.rand(2,sample_count,generator=rng,dtype=torch.float64)<q[:,None]).double()
    empirical = labels.mean(1)
    rows = []
    for initial in ([.5,.5],[.2,.4],[.8,.3]):
        for kind in ('analytic_expected','sampled_labels_logistic_network'):
            for arm in ('C0','C1','C2'):
                # A 2-input one-hot linear sigmoid network; no hidden oracle.
                net = torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
                with torch.no_grad(): net.weight.copy_(torch.logit(torch.tensor(initial,dtype=torch.float64))[None])
                optimizer = torch.optim.Adam(net.parameters(),lr=.05)
                for _ in range(steps):
                    p = net(torch.eye(2,dtype=torch.float64)).flatten().sigmoid()
                    target = q if kind == 'analytic_expected' else empirical
                    # Paired-label empirical mean is exactly the full-batch
                    # expectation of each sampled singleton ES (cross linear in y).
                    loss = objective(p,target,arm,weight)
                    optimizer.zero_grad();loss.backward();optimizer.step()
                p = net(torch.eye(2,dtype=torch.float64)).flatten().sigmoid().detach()
                rows.append(dict(initial=initial,kind=kind,arm=arm,probabilities=p.tolist(),
                                 conditional_probability_mae=(p-q).abs().mean().item(),
                                 marginal_probability_error=abs(p.mean().item()-.5)))
    return dict(true_conditional=q.tolist(),empirical_conditional=empirical.tolist(),
                labels_per_input=sample_count,labels_hash=canonical_hash(labels.int().tolist()),
                label_seed=2103,lambda_group=weight,steps=steps,rows=rows,
                analytic_optima=dict(C0=q.tolist(),C1=((q+weight*.5)/(1+weight)).tolist(),C2=q.tolist()))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    with args.output.open('x') as stream: json.dump(run_toy(),stream,indent=2)
