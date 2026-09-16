import argparse,sys,time
import numpy as np
import torch
from .common import *
from generator_reward_nft.public import Generator
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow
from hirp.phase24 import LEGACY,verify_files,normalization
from hirp.task_phase24 import check_prediction

def main(arm):
 configure_flow();torch.set_num_threads(4);sys.path.insert(0,str(LEGACY))
 from framework.metrics.FRC import _func
 from framework.metrics.S_MSE import compute_s_mse
 from framework.metrics.FRVar import compute_FRVar
 from framework.metrics.TLCC import _func as tlcc
 from framework.utils.compute_metrics import _func as mae
 manifest=read(ROOT/'manifest.json');normalization(manifest);target_index=read(ROOT/'targets.json');assert target_index['manifest_sha256']==sha(ROOT/'manifest.json')
 ck=manifest['checkpoints'][arm];verify_files([ck]);gen=Generator()
 if arm!='P2':
  saved=torch.load(ck['path'],map_location='cpu',weights_only=True);assert saved['step']==1000 and saved['arm']==arm;gen.model.base.velocity.load_state_dict(saved['student'])
 gen.model.eval().requires_grad_(False);mean=torch.from_numpy(np.load('external/FaceVerse/mean_face.npy')).float();std=torch.from_numpy(np.load('external/FaceVerse/std_face.npy')).float().clamp_min(1e-8)
 dest=ROOT/'exports'/arm;dest.mkdir(parents=True,exist_ok=True);rows=[]
 for i,s in enumerate(manifest['sources']):
  side=dest/f'{i:04d}.json'
  if side.exists():
   row=read(side);assert row['checkpoint_sha256']==ck['sha256'];verify_files([row['export']]);rows.append(row);continue
  verify_files(list(s['files'].values()));name=s['clip_id'];n=s['length'];idx=s['dev80_index'];nonnull=False

  streams={k:torch.from_numpy(np.load(v['path'])).float() for k,v in s['files'].items()};face=streams['speaker_3dmm'];streams['speaker_3dmm']=((face[:,0] if face.ndim==3 else face)-mean)/std
  events=[None for start in range(0,n,750)]
  noise=recording_noise(gen.model.base.config,name,n).cuda().double();pred=gen.sample([streams[k] for k in ('speaker_audio','speaker_emotion','speaker_3dmm')],s['frame_pts'],events,noise).float().cpu();check_prediction(pred,n)
  path=dest/f'{i:04d}.npy';tmp=path.with_suffix('.tmp')
  with tmp.open('wb') as f:np.save(f,pred.numpy())
  tmp.replace(path);export=dict(path=str(path),sha256=sha(path))
  t=target_index['files'][i];verify_files([t]);target=torch.from_numpy(np.load(t['path']));assert target.shape==pred.shape==(10,n,25)
  metrics=dict(MAE=float(mae(target,pred)),FRC=float(_func(target,pred)),S_MSE=float(compute_s_mse([pred])),FRVar=float(compute_FRVar([pred])),temporal_S_MSE=float(compute_s_mse([pred-pred.mean(1,keepdim=True)])),TLCC=float(tlcc(pred,torch.from_numpy(np.load(s['files']['speaker_emotion']['path'])).float())));assert all(np.isfinite(v) for v in metrics.values())
  row=dict(index=i,clip_id=name,checkpoint_sha256=ck['sha256'],export=export,metrics=metrics,dev80=idx is not None,semantic_nonnull=nonnull);write(side,row);rows.append(row);write(ROOT/(arm+'_status.json'),dict(completed=i+1,total=manifest['population'],stage='generation_and_FRC',updated_unix=time.time()));print(arm,i+1,'/',manifest['population'],flush=True)
 assert len(rows)==manifest['population']
 groups={'full_local_TEST':rows,'original_speaker_direction':[r for r in rows if r['clip_id'].startswith('speaker/')],'reverse_direction':[r for r in rows if r['clip_id'].startswith('listener/')] }
 result={k:dict(count=len(v),**{m:float(np.mean([r['metrics'][m] for r in v])) for m in ['FRC','S_MSE','FRVar','temporal_S_MSE','TLCC','MAE']}) for k,v in groups.items() if v}
 write(ROOT/(arm+'_RESULTS.json'),dict(checkpoint=ck,manifest_sha256=sha(ROOT/'manifest.json'),source_only=True,K=10,results=result,rows=rows,aggregation='arithmetic mean of official per-source FRC/S-MSE/FRVar; original FRC sums best-target CCC over K10'))
 print(arm,result,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
