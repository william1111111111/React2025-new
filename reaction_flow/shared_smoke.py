"""Actual FP32 T750/B4/K4 gradient and exact new-stage resume smoke."""
import json
from pathlib import Path
import torch
from .train_shared import train,ROOT
from .train import configure_flow,write
from mam_refine.train import tree_hash

def main():
    configure_flow()
    train('G1-shared',2,label='smoke_full')
    train('G1-shared',1,label='smoke_split')
    train('G1-shared',2,resume=ROOT/'smoke_split/attempt_000/checkpoints/step_014001.pt',label='smoke_resume')
    a=torch.load(ROOT/'smoke_full/attempt_000/checkpoints/step_014002.pt',map_location='cpu',weights_only=True)
    b=torch.load(ROOT/'smoke_resume/attempt_000/checkpoints/step_014002.pt',map_location='cpu',weights_only=True)
    error=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    assert error==0 and tree_hash(a['optimizer'])==tree_hash(b['optimizer'])
    assert a['rows'][-1]['task']>0 and a['rows'][-1]['gradient_norm']>0
    write(ROOT/'resume_regression.json',dict(max_parameter_error=error,optimizer_equal=True,steps=2,full_shape='T750 B4 K4 Euler16',condition_frozen=True))
if __name__=='__main__':main()
