import numpy as np
from reward_signal_v2.rewards import affinity,coverage_gains
from .common import read,OLD,V2

def components(s,ref,n,target_lengths,dual):
    scales=read(OLD/'scales.json');metric=read(V2/'GEOMETRY_STRATIFIED.json')
    # Dynamic empirical envelopes are diagnostic until scenario-specific validity is established.
    # Penalize domain invalidity only; no claim that old range-max motion gates are realistic.
    good=s['domain_valid'];refgood=ref['domain_valid']
    c=s['ccc'].max(1);d=s['dtw'].min(1);cr=ref['ccc'].max(1).mean();dr=ref['dtw'].min(1).mean()
    q=dual[0]*(c-cr)/scales['c']-dual[1]*(d-dr)/scales['d']-dual[2]*(~good)
    mask=np.stack([np.arange(96)//24<min(4,n,nn) for nn in target_lengths])
    sim,_=affinity(s['phi'],s['target_phi'],np.array(metric['scales']['phi']),mask,metric['h2'])
    eligible=(s['ccc']>=np.quantile(ref['ccc'].max(1),.1))&(s['dtw']<=np.quantile(ref['dtw'].min(1),.9))&good[:,None]
    b=coverage_gains(sim*eligible)
    violation=np.array([(cr-c.mean())/scales['c'],(d.mean()-dr)/scales['d'],float((~good).mean()-(~refgood).mean())])
    return q,b,eligible.any(1),violation,dict(mean_CCC=float(c.mean()),mean_DTW=float(d.mean()),coverage=float((sim*eligible).max(0).mean()),eligible_pair_rate=float(eligible.mean()),nontrivial_gain_fraction=float((b>1e-6).mean()),bad_rate=float((~good).mean()))

def weights(q,b,supported,arm,cal):
    raw=q+(cal['alpha']*b if arm=='N1-quality-coverage' else 0)
    centered=raw-raw.mean();r=.5+.5*np.clip(centered/cal['Z'][arm],-1,1)
    r[~supported]=np.minimum(r[~supported],.5)
    return r
