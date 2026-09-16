"""CPU-only official metrics, independently parallel to GPU generation."""
import sys,time
import numpy as np
import torch
from .common import *
from hirp.phase24 import LEGACY,verify_files

def score(job):
 torch.set_num_threads(1);sys.path.insert(0,str(LEGACY))
 from framework.metrics.FRC import _func
 from framework.metrics.S_MSE import compute_s_mse
 from framework.metrics.FRVar import compute_FRVar
 from framework.metrics.TLCC import _func as tlcc
 from framework.utils.compute_metrics import _func as mae
 s,export,target,ck,i,side=job;t=time.perf_counter();verify_files([export,target]);pred=torch.from_numpy(np.load(export['path']));y=torch.from_numpy(np.load(target['path']));assert pred.shape==y.shape==(10,s['length'],25)
 metrics=dict(MAE=float(mae(y,pred)),FRC=float(_func(y,pred)),S_MSE=float(compute_s_mse([pred])),FRVar=float(compute_FRVar([pred])),temporal_S_MSE=float(compute_s_mse([pred-pred.mean(1,keepdim=True)])),TLCC=float(tlcc(pred,torch.from_numpy(np.load(s['files']['speaker_emotion']['path'])).float())))
 assert all(np.isfinite(v) for v in metrics.values())
 row=dict(index=i,clip_id=s['clip_id'],checkpoint_sha256=ck['sha256'],export=export,metrics=metrics,dev80=False,semantic_nonnull=False,cpu_metric_seconds=time.perf_counter()-t);write(side,row);return row
