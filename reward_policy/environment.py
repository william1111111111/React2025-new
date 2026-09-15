"""TRAIN-only reward environment; exact DTW and fixed source/target/block episodes."""
import time
from dataclasses import replace
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tslearn.metrics import dtw
from mam_target.losses import ccc25
from mode_supervision.data import ModeData
from mode_supervision.plans import basis,summarize
from semantic_supervision.bert_experiments.common import Sources
from reaction_flow.export import recording_noise
from .generator import Generator

class TrainEnvironment:
    def __init__(self):
        self.generator=Generator();self.data=ModeData(self.generator.model.base);self.semantic=Sources('train');self.pool=ThreadPoolExecutor(max_workers=4)
        assert self.data.pool.split=='train'
    def episode(self,rec):
        idx=rec['source_indices'][0];r=self.data.rows[idx];pts=r['frame_pts'];start=rec['block_indices'][0]*750;n=min(750,len(pts)-start);name=r['clip_id']
        events=[self.semantic.crop(name,s,min(750,len(pts)-s))[0] if name in self.semantic.rows else None for s in range(0,len(pts),750)]
        hs,ms,c=self.generator.features(self.data.source(idx),pts,events)
        ids=self.data.refs(rec,0);ys=[self.data.aligned(idx,ref,j==0)[start:start+n] for j,ref in enumerate(ids)]
        assert min(map(len,ys))>=2
        return dict(rec=rec,pts=pts,start=start,n=n,context=c,h=hs[start//750],mask=ms[start//750],targets=ys,target_ids=ids,basis=basis(pts).cuda(),clip_id=name,semantic_nonnull=any(events))
    def noise(self,e,k,kind):
        cfg=self.generator.model.base.config
        # Original generator algorithm and candidate keys, with independent TRAIN bank seeds.
        seed={'reference':9100123,'action':9200123,'reachability':9300123}[kind]
        return recording_noise(replace(cfg,evaluation_seed=seed),e['clip_id']+'|episode'+str(e['rec']['step']),len(e['pts']),k).cuda().double()
    def generate(self,e,noise):return self.generator.block(e['h'],e['mask'],noise,e['start'],e['n']).float().cpu()
    def scores(self,e,pred):
        tick=time.time();targets=e['targets'];ccc=np.array([[float(ccc25(p[:len(y)],y)) for y in targets] for p in pred]);pairs=[(p.numpy(),y.numpy()) for p in pred for y in targets]
        def distance(pair):
            p,y=pair;n=min(len(p),len(y));return sum(w*dtw(p[:n,a:b].astype(np.float32),y[:n,a:b].astype(np.float32)) for a,b,w in ((0,15,1/15),(15,17,1),(17,25,1/8)))
        distances=np.array(list(self.pool.map(distance,pairs))).reshape(len(pred),len(targets))
        def phi(y):
            pts=e['pts'][e['start']:e['start']+len(y)]
            with torch.no_grad():a,_=summarize(self.generator.model.base.transform(y.cuda().double()),pts)
            return a.flatten().cpu().numpy()
        shape=np.stack([phi(y) for y in pred]);target_shape=np.stack([phi(y) for y in targets])
        speed=np.array([float(y.diff(dim=0).abs().max()) if len(y)>1 else 0 for y in pred]);accel=np.array([float(y.diff(dim=0).diff(dim=0).abs().max()) if len(y)>2 else 0 for y in pred])
        valid=np.array([bool(torch.isfinite(y).all() and (y[:,:15]>=0).all() and (y[:,:15]<=1).all() and (y[:,15:17].abs()<=1).all() and (y[:,17:]>=0).all() and torch.allclose(y[:,17:].sum(-1),torch.ones(len(y)),atol=1e-5)) for y in pred])
        return dict(ccc=ccc,dtw=distances,phi=shape,target_phi=target_shape,speed=speed,accel=accel,domain_valid=valid,metric_seconds=time.time()-tick)
