"""Batched native chunks and bounded asynchronous CPU metric workers."""
import argparse,time,multiprocessing,concurrent.futures
import numpy as np
import torch
from .common import *
from .batched import BatchedGenerator
from .metric_worker import score
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow
from hirp.phase24 import verify_files,normalization
from hirp.task_phase24 import check_prediction

def main(arm):
 configure_flow();torch.set_num_threads(4);manifest=read(ROOT/'manifest.json');normalization(manifest);ti=read(ROOT/'targets.json');assert ti['manifest_sha256']==sha(ROOT/'manifest.json');assert read(ROOT/'BATCHING_BENCHMARK.json')['prediction_max_abs_error']<=1e-6
 ck=manifest['checkpoints'][arm];verify_files([ck]);gen=BatchedGenerator()
 if arm!='P2':
  saved=torch.load(ck['path'],map_location='cpu',weights_only=True);assert saved['step']==1000 and saved['arm']==arm;gen.model.base.velocity.load_state_dict(saved['student'])
 gen.model.eval().requires_grad_(False);mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
 dest=ROOT/'exports'/arm;dest.mkdir(parents=True,exist_ok=True);rows={};pending={};generated=0;tick=time.time()
 def status():write(ROOT/(arm+'_status.json'),dict(completed=len(rows),generated=generated,total=manifest['population'],stage='batched_generation_async_metrics',pending_metrics=len(pending),updated_unix=time.time(),seconds=time.time()-tick))
 def drain(block=False):
  if not pending:return
  done,_=concurrent.futures.wait(pending,timeout=None if block else 0,return_when=concurrent.futures.FIRST_COMPLETED)
  for f in done:
   row=f.result();rows[row['index']]=row;pending.pop(f)
  status()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for i,s in enumerate(manifest['sources']):
   side=dest/f'{i:04d}.json'
   if side.exists():
    row=read(side);assert row['checkpoint_sha256']==ck['sha256'];verify_files([row['export']]);rows[i]=row;generated+=1;continue
   gen_side=dest/f'{i:04d}_generation.json';identity=dict(checkpoint_sha256=ck['sha256'],manifest_sha256=sha(ROOT/'manifest.json'),source=i)
   if gen_side.exists():
    cached=read(gen_side);assert cached['identity']==identity;export=cached['export'];verify_files([export])
   else:
    verify_files(list(s['files'].values()));n=s['length'];streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std;noise=recording_noise(gen.model.base.config,s['clip_id'],n).cuda().double();start=time.perf_counter()
    pred=gen.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],s['frame_pts'],[None for _ in range(0,n,750)],noise).float().cpu();check_prediction(pred,n)
    path=dest/f'{i:04d}.npy';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,pred.numpy())
    tmp.replace(path);export=dict(path=str(path),sha256=sha(path));write(gen_side,dict(identity=identity,export=export,generation_seconds=time.perf_counter()-start))
   generated+=1;f=pool.submit(score,(s,export,ti['files'][i],ck,i,side));pending[f]=i;drain();print(arm,'generated',generated,'metrics',len(rows),'/',manifest['population'],flush=True)
   while len(pending)>=12:drain(True)
  while pending:drain(True)
 ordered=[rows[i] for i in range(manifest['population'])];groups={'full_local_TEST':ordered,'original_speaker_direction':[r for r in ordered if r['clip_id'].startswith('speaker/')],'reverse_direction':[r for r in ordered if r['clip_id'].startswith('listener/')]}
 result={k:dict(count=len(v),**{m:float(np.mean([r['metrics'][m] for r in v])) for m in ['FRC','S_MSE','FRVar','temporal_S_MSE','TLCC','MAE']}) for k,v in groups.items() if v}
 write(ROOT/(arm+'_RESULTS.json'),dict(checkpoint=ck,manifest_sha256=sha(ROOT/'manifest.json'),source_only=True,K=10,results=result,rows=ordered,aggregation='unchanged mean of official per-source metrics',execution='two native blocks x K10 per rollout batch; four asynchronous CPU metric workers'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
