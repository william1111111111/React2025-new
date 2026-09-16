"""Fixed 64 TRAIN groups, independent cached P2 bank; no DEV input."""
import time
import numpy as np
import torch
from reaction_flow.train import configure_flow
from reward_policy.pilot import arrays,serial
from .model import Models
from .rewards import components
from .common import *

def main():
    configure_flow();torch.set_num_threads(4);m=Models();records=read(OLD/'episodes.json')['calibration'];rows=[]
    for i,rec in enumerate(records):
        p=OUT/'calibration'/f'{i:03d}.json'
        if p.exists():rows.append(read(p));continue
        e=m.env.episode(rec);noise=m.env.noise(e,10,'action');clean,pred=m.generate(e,noise,m.ref)
        original=m.env.generate(e,noise);assert torch.equal(pred,original)
        s=m.env.scores(e,pred);ref=arrays(read(OLD/'calibration'/(str(rec['step'])+'.json'))['scores'])
        q,b,supported,violation,stats=components(s,ref,e['n'],[len(y) for y in e['targets']],np.ones(3))
        dynamics={}
        for label,ys in [('real',e['targets']),('parent',list(pred))]:
            dynamics[label]={}
            for group,lo,hi in [('AU',0,15),('VA',15,17),('expression',17,25)]:
                ds={}
                for order,name in [(1,'speed'),(2,'acceleration')]:
                    values=torch.cat([y[:,lo:hi].diff(n=order,dim=0).abs().flatten() for y in ys if len(y)>order]).numpy()
                    ds[name]={str(t):float(np.quantile(values,t)) for t in [.95,.99,1.]}
                dynamics[label][group]=ds
        row=dict(record=rec,q=q.tolist(),b=b.tolist(),supported=supported.tolist(),stats=stats,dynamics=dynamics,scores=serial(s),reference_path=str(OLD/'calibration'/(str(rec['step'])+'.json')),reference_sha256=sha(OLD/'calibration'/(str(rec['step'])+'.json')),native_prediction_exact=True,new_candidate_rollouts=10,native_identity_check_rollouts=10,reference_rollouts_reused=10)
        write(p,row);rows.append(row);write(OUT/'calibration_status.json',dict(completed=i+1,total=64));print('calibration',i+1,flush=True)
    q=np.array([r['q'] for r in rows]);b=np.array([r['b'] for r in rows]);assert np.any(b>1e-6),'Coverage absent: do not fabricate variance'
    alpha=float(np.clip(q.std()/max(b.std(),1e-3),.1,10))
    # Fixed pooled within-group centered scales. Global std with 1e-3 floor, no per-group normalization.
    Z={a:float(max((v-v.mean(1,keepdims=True)).std(),1e-3)) for a,v in zip(ARMS,[q,q+alpha*b])}
    write(OUT/'CALIBRATION.json',dict(alpha=alpha,Z=Z,q_std=float(q.std()),b_std=float(b.std()),q_mean=float(q.mean()),b_mean=float(b.mean()),scaling='pooled TRAIN within-group centered std; floor1e-3; fixed before formal rounds',r_is_probability=False,records=64,new_candidate_rollouts=640,native_identity_check_rollouts=640,reference_rollouts_reused=640,dynamics='per-calibration-group q95/q99/max for real and P2, diagnostic not natural-motion guarantee',geometry_sha256=sha(V2/'GEOMETRY_STRATIFIED.json')))
if __name__=='__main__':main()
