import sys,time,json
import numpy as np
import torch
from .common import *
from .batched import BatchedGenerator
from generator_reward_nft.public import Generator
from reaction_flow.train import configure_flow
from reaction_flow.export import recording_noise
from hirp.phase24 import LEGACY

def main():
 configure_flow();torch.set_num_threads(4);sys.path.insert(0,str(LEGACY))
 from framework.metrics.FRC import _func
 from framework.metrics.TLCC import _func as tlcc
 from framework.metrics.S_MSE import compute_s_mse
 from framework.metrics.FRVar import compute_FRVar
 from framework.utils.compute_metrics import _func as mae
 gen=BatchedGenerator();manifest=read(ROOT/'manifest.json');s=manifest['sources'][0];n=s['length'];streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8);streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std;xs=[streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')];events=[None for _ in range(0,n,750)];noise=recording_noise(gen.model.base.config,s['clip_id'],n).cuda().double();results={}
 torch.cuda.synchronize();t=time.perf_counter();old=Generator.sample(gen,xs,s['frame_pts'],events,noise).float().cpu();torch.cuda.synchronize();results['original_generation_seconds']=time.perf_counter()-t
 t=time.perf_counter();new=gen.sample(xs,s['frame_pts'],events,noise).float().cpu();torch.cuda.synchronize();results['batched_generation_seconds']=time.perf_counter()-t
 results['prediction_max_abs_error']=float((old-new).abs().max());results['prediction_exact_equal']=torch.equal(old,new);torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-6)
 target=torch.from_numpy(np.load(read(ROOT/'targets.json')['files'][0]['path']));results['cpu_metric_seconds']={};results['metric_differences']={}
 for key,fn in [('FRC',lambda y:_func(target,y)),('TLCC',lambda y:tlcc(y,streams['speaker_emotion'])),('S_MSE',lambda y:compute_s_mse([y])),('FRVar',lambda y:compute_FRVar([y])),('MAE',lambda y:mae(target,y))]:
  t=time.perf_counter();a=float(fn(old));results['cpu_metric_seconds'][key]=time.perf_counter()-t;b=float(fn(new));results['metric_differences'][key]=abs(a-b);assert abs(a-b)<1e-6,(key,a,b)
 results['source']=s['clip_id'];results['frames']=n;results['peak_GPU_GiB']=torch.cuda.max_memory_allocated()/2**30;write(ROOT/'BATCHING_BENCHMARK.json',results);print(results,flush=True)
if __name__=='__main__':main()
