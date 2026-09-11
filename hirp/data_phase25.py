"""Occurrence-specific source crops and independent reference crops, raw cache only."""
import hashlib
import json
import random
from functools import lru_cache
import torch
from torch.utils.data import default_collate
from .paired_data import PairedReactionDataset
from .session_data import SessionPopulation
from .group_features import DescriptorScaler,reaction_descriptor
from .train_phase1 import to_device
from .phase15_audit import canonical_hash,sha256_file

class RawCachedDataset(PairedReactionDataset):
    @staticmethod
    @lru_cache(maxsize=48)
    def _load(path):
        return PairedReactionDataset._load(path)

def reference_crop(pool,ref,seed,step):
    x=pool.dataset._load(pool.reference_paths[ref])
    payload=f'reference|{seed}|{step}|{ref}'.encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8],'big'))
    start=rng.randint(0,max(0,len(x)-pool.clip_length))
    n=min(pool.clip_length,len(x)-start)
    if n<1:raise ValueError('empty reference')
    y=x.new_zeros(pool.clip_length,25);y[:n]=x[start:start+n]
    return y,torch.tensor(n),start

class RealData:
    def __init__(self,manifest,cfg,arm,device):
        self.pool=SessionPopulation('data','train',cfg.T)
        self.pool.dataset=RawCachedDataset('data','train',cfg.T,crop_mode='random',seed=cfg.sampler_seed)
        self.scaler=DescriptorScaler.load(manifest['scaler_path']).to(device)
        self.arm=arm;self.device=device
        self.invalid_pairs=0
    def batch(self,record):
        items=[]
        for index,occurrence in zip(record['source_indices'],record['crop_occurrences']):
            self.pool.dataset.set_epoch(occurrence)
            try:items.append(self.pool.dataset[index])
            except ValueError as exc:
                if 'no paired overlap' in str(exc):self.invalid_pairs+=1
                raise
        batch=to_device(default_collate(items),self.device)
        if self.arm=='C0':return batch,None
        refs=[]
        with torch.no_grad():
            for ref in record['reference_ids']:
                y,n,_=reference_crop(self.pool,ref,record['crop_seed'],record['reference_crop_step'])
                refs.append(self.scaler(*reaction_descriptor(y.to(self.device),n.to(self.device))))
        return batch,tuple(torch.stack([r[j] for r in refs]) for j in (0,1))

def fit_scaler(pool,path):
    """Four fixed train-only uniform crops per unique reference, session-weighted."""
    if pool.split!='train':raise ValueError('train only')
    features=[];masks=[];weights=[];crops=[];files=[]
    N=sum(len(s) for s in pool.sources.values())
    for session in sorted(pool.sources):
        refs=pool.references[session]
        for ref in refs:
            files.append(dict(path=str(pool.reference_paths[ref]),sha256=sha256_file(pool.reference_paths[ref])))
            for draw in range(4):
                y,n,start=reference_crop(pool,ref,25000,draw)
                f,m=reaction_descriptor(y,n);features.append(f.double());masks.append(m)
                weight=len(pool.sources[session])/N/len(refs)/4;weights.append(weight)
                crops.append(dict(reference_id=ref,draw=draw,seed=25000,crop_start=start,length=int(n),weight=weight))
    x=torch.stack(features);mask=torch.stack(masks);w=torch.tensor(weights,dtype=torch.float64)[:,None]*mask
    mean=(x*w).sum(0)/w.sum(0);var=((x-mean).square()*w).sum(0)/w.sum(0)
    state=dict(split='train',clip_length=pool.clip_length,crop_mode='independent_uniform',seed=25000,
        mean_train=mean.tolist(),std_train=var.sqrt().tolist(),valid_counts=mask.sum(0).tolist(),std_floor=1e-4,
        weighting='p(session)=n_source/N; uniform unique references; four uniform crop draws each',
        source_files=files,source_data_hash=canonical_hash(files),reference_crops=crops)
    with path.open('x') as f:json.dump(state,f,indent=2,allow_nan=False)
