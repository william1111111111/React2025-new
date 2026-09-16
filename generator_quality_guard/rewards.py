import numpy as np
from generator_reward_nft.rewards import components as prior_components

def components(s,ref,n,lengths,dual):
 q,b,support,violation,stats=prior_components(s,ref,n,lengths,np.ones(3))
 # q already uses fixed parent mean C0/D0. Preserve it; no within-group subtraction.
 stats.update(candidate_c=s['ccc'].max(1).tolist(),candidate_d=s['dtw'].min(1).tolist(),C0=float(ref['ccc'].max(1).mean()),D0=float(ref['dtw'].min(1).mean()),violation_C=float(violation[0]),violation_D=float(violation[1]),coverage_group_feasible=bool(violation[0]<=0 and violation[1]<=0))
 stats.update(v_C_raw=stats['C0']-stats['mean_CCC'],v_D_raw=stats['mean_DTW']-stats['D0'])
 return q,b,support,violation,stats

def weights(q,b,support,arm,cal,feasible=True):
 r=.5+.5*np.tanh(q/cal['Z_Q'])
 if arm=='Q1-coverage-guarded' and feasible:r=r+.5*np.tanh(b/cal['Z_B'])*support
 r=np.clip(r,0,1);r[~support]=np.minimum(r[~support],.5)
 return r
