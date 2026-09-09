"""Exact optimizer/log/RNG comparison after cold-start Python RNG correction."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from .train_phase21 import configure
from .train_phase22 import TrainConfig,RealData,train,write


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:4');a=p.parse_args();configure()
    root=a.root;old=root/'resume_smoke';out=root/'rng_fix_resume';out.mkdir(exist_ok=False)
    m=json.loads((old/'manifest.json').read_text());config=m['config'];config['fixed_checkpoints']=tuple(config['fixed_checkpoints']);cfg=TrainConfig(**config)
    records=json.loads((old/'schedule.json').read_text())['records'];noise=torch.from_numpy(np.load(old/'noise.npy'));results={}
    for arm in ('C0','C1','C2'):
        data=RealData(m,cfg,arm,a.device)
        full,model,opt=train(out/('full_'+arm),cfg,arm,records,noise,data,m,a.device);del model,opt
        stop,model,opt=train(out/('split_'+arm),cfg,arm,records,noise,data,m,a.device,stop_after=4);del model,opt
        resumed,model,opt=train(out/('split_'+arm),cfg,arm,records,noise,data,m,a.device,resume=stop['checkpoints'][-1]['path']);del model,opt
        x=torch.load(full['checkpoints'][-1]['path'],weights_only=True,map_location='cpu');y=torch.load(resumed['checkpoints'][-1]['path'],weights_only=True,map_location='cpu')
        checks={k:equal(x[k],y[k]) for k in ('model','optimizer','training_rows','rng','global_step','consumed_hashes')}
        # Compare tensor training results with the already archived original smoke.
        previous=torch.load(old/('full_'+arm)/'attempt_000/checkpoints/step_000008.pt',weights_only=True,map_location='cpu')
        checks['unchanged_from_original_model']=equal(x['model'],previous['model'])
        checks['unchanged_from_original_optimizer']=equal(x['optimizer'],previous['optimizer'])
        assert all(checks.values()),checks
        results[arm]=checks
        del data,x,y,previous
    write(out/'verification.json',results)


if __name__=='__main__':main()
