"""Evaluation-only covariance and lag correlation; no fitting on validation."""
import torch

LAGS=(1,4,16)
VERSION='raw-units-cov-and-acf-lags1-4-16-v1'


def auxiliary(pred,lengths):
    # Fixed raw units avoid learned evaluation scales. Population covariance,
    # signed autocorrelation numerator / variance, zero for constant channels.
    results=[]
    for i,n in enumerate(lengths.tolist()):
        x=pred[i,:,:n].float();z=x-x.mean(1,keepdim=True)
        cov=torch.einsum('ktc,ktd->kcd',z,z)/n
        ac=[]
        for lag in LAGS:
            numerator=(z[:,lag:]*z[:,:-lag]).mean(1) if n>lag else z.new_zeros(z.shape[0],25)
            denominator=z.square().mean(1)
            ac.append(torch.where(denominator>0,numerator/denominator.clamp_min(1e-12),torch.zeros_like(numerator)))
        results.append(dict(covariance=cov.mean(0).tolist(),autocorrelation=torch.stack(ac).mean(1).tolist()))
    return results


def exact_decomposition(probabilities,weights,q,support):
    """Finite distributions with exact Euclidean distance, independent draws."""
    d=torch.cdist(support,support)
    es=lambda p:p@d@q-.5*(p@d*p).sum(-1)
    pbar=weights@probabilities
    left=(weights*es(probabilities)).sum();marginal=es(pbar)
    self_dist=(probabilities@d*probabilities).sum(-1)
    energy=2*probabilities@d@probabilities.T-self_dist[:,None]-self_dist[None,:]
    extra=.25*(weights[:,None]*weights[None,:]*energy).sum()
    return left,marginal,extra
