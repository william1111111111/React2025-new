"""Sixteen fixed full TRAIN recordings; official Processor and exact native metrics."""
import sys,time,random,concurrent.futures,multiprocessing
from dataclasses import replace
import numpy as np
import torch
from .common import *
from generator_reward_nft.model import Models
from reaction_flow.train import configure_flow
from reaction_flow.export import recording_noise
from hirp.phase24 import LEGACY,official_selection,verify_files

def prepare():
 root=OUT/'monitor';root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists():return
 sys.path.insert(0,str(LEGACY));from regnn.eval_conditional_regnn_official_test import initialize_official_hydra_runtime
 initialize_official_hydra_runtime(LEGACY)
 from dataset.react_2025 import ReactionDataset
 from framework.modules.post_processor import Processor
 ds=ReactionDataset(root_dir=str(Path('data').resolve()),split='train',clip_length=750,load_video_s=False,load_video_l=False);lookup={str(p):i for i,p in enumerate(ds.speaker_path_list)};rows=read('runs/reaction_flow/mode_supervision_v1/cohort.json');sessions=sorted({r['clip_id'].split('/')[1] for r in rows});selected=[]
 for j in range(16):
  session=sessions[j*len(sessions)//16];options=sorted([r for r in rows if r['clip_id'].split('/')[1]==session and min(r['source_length'],r['paired_length'])>=2],key=lambda r:(r['source_length'],r['clip_id']));pos=min(len(options)-1,int(((j%4)+.5)/4*len(options)));selected.append(options[pos])
 proc=Processor(cfg_dir=str(LEGACY),ckpt_dir=str(LEGACY/'pretrained_models/post_processor'),device=torch.device('cuda:0'),clip_len_test=1000,num_preds=10);cases=[]
 for j,row in enumerate(selected):
  name=row['clip_id'];i=lookup[name];refs=official_selection(ds.listener_path_list[i],ds.gt_path_list[i],random.Random(9571000+j));targets=[]
  for ref in refs:
   p=Path('data/train/facial-attributes')/ref.with_suffix('.npy');targets.append(dict(path=str(p),sha256=sha(p)))
  raw=[torch.from_numpy(np.load(p['path'])).float() for p in targets];torch.manual_seed(9572000+j)
  with torch.no_grad():y=proc.forward([torch.zeros(10,row['source_length'],25)],[raw])[0]
  assert y.shape==(10,row['source_length'],25) and torch.isfinite(y).all()
  path=root/f'target_{j:02d}.npy'
  with path.open('wb') as f:np.save(f,y.cpu().numpy())
  files={}
  for field,folder in [('speaker_audio','audio-features'),('speaker_emotion','facial-attributes'),('speaker_3dmm','coefficients')]:
   p=Path('data/train')/folder/(name+'.npy');files[field]=dict(path=str(p),sha256=sha(p))
  cases.append(dict(index=row['index'],clip_id=name,length=row['source_length'],frame_pts=row['frame_pts'],targets=targets,processed=dict(path=str(path),sha256=sha(path)),files=files));print('monitor targets',j+1,flush=True)
 write(root/'manifest.json',dict(split='train',selection='16 evenly indexed sessions; cycling within-session length quartiles, no model scores',cases=cases,K=10,processor_seed_base=9572000,noise_seed=9570123,scope='TRAIN control set, not independent validation',processor_sha256=sha(LEGACY/'framework/modules/post_processor.py')))

def score_case(args):
 predfile,targetfile,j=args;torch.set_num_threads(1);sys.path.insert(0,str(LEGACY))
 from framework.metrics.FRC import concordance_correlation_coefficient
 from tslearn.metrics import dtw
 from framework.metrics.S_MSE import compute_s_mse
 from reaction_flow.dynamics import decomposition
 p=torch.from_numpy(np.load(predfile));y=torch.from_numpy(np.load(targetfile));n=len(p[0]);C=np.array([[concordance_correlation_coefficient(t.numpy(),v.numpy())[0] for t in y] for v in p]);D=[];tick=time.time()
 for v in p:
  D.append([sum(w*float(dtw(v[:,a:b].numpy().astype(np.float32),t[:,a:b].numpy().astype(np.float32))) for a,b,w in [(0,15,1/15),(15,17,1),(17,25,1/8)]) for t in y])
 D=np.array(D);groups={}
 for group,a,b in [('AU',0,15),('VA',15,17),('expression',17,25)]:
  total,parts=decomposition(p[...,a:b].double());groups[group]=dict(total=float(total),**{k:float(v) for k,v in parts.items()},speed_q99=float(torch.quantile(p[...,a:b].diff(dim=1).abs().flatten(),.99)),acceleration_q99=float(torch.quantile(p[...,a:b].diff(n=2,dim=1).abs().flatten(),.99)) if n>2 else 0.)
 return dict(index=j,FRC=float(C.max(1).sum()),FRD=float(D.min(1).sum()),S_MSE=float(compute_s_mse([p])),FRVar=float(p.var(1).mean()),target_side_best_CCC_mean=float(C.max(0).mean()),target_side_best_DTW_mean=float(D.min(0).mean()),candidate_quality_tail={str(q):float(np.quantile(C.max(1),q)) for q in [0,.1,.25,.5]},CCC=C.tolist(),DTW=D.tolist(),native_groups=groups,exact_pairs=100,seconds=time.time()-tick)

def evaluate(models,velocity,label):
 manifest=read(OUT/'monitor/manifest.json');root=OUT/'monitor'/label;root.mkdir(parents=True,exist_ok=True);vh=digest(velocity);summary=root/'summary.json'
 if summary.exists():
  prior=read(summary);assert prior['velocity_sha256']==vh;return prior
 tasks=[]
 for j,s in enumerate(manifest['cases']):
  verify_files(list(s['files'].values())+s['targets']+[s['processed']]);p=root/f'{j:02d}.npy';side=root/f'{j:02d}_export.json'
  if side.exists():meta=read(side);assert meta['velocity_sha256']==vh;verify_files([meta])
  else:
   idx=s['index'];n=s['length'];events=[models.env.semantic.crop(s['clip_id'],start,min(750,n-start))[0] if s['clip_id'] in models.env.semantic.rows else None for start in range(0,n,750)];hs,ms,_=models.env.generator.features(models.env.data.source(idx),s['frame_pts'],events);cfg=replace(models.env.generator.model.base.config,evaluation_seed=9570123);noise=recording_noise(cfg,s['clip_id']+'|fixed_TRAIN_quality_monitor',n,10).cuda().double();pred=[]
   for h,m,start in zip(hs,ms,range(0,n,750)):
    _,y=models.generate(dict(h=h,mask=m,start=start,n=min(750,n-start)),noise,velocity);pred.append(y)
   y=torch.cat(pred,1);tmp=p.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,y.numpy())
   tmp.replace(p);write(side,dict(path=str(p),sha256=sha(p),velocity_sha256=vh))
  result=root/f'{j:02d}_metrics.json'
  if not result.exists():tasks.append((str(p),s['processed']['path'],j))
  print(label,'generated',j+1,'/16',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as ex:
  for row in ex.map(score_case,tasks):write(root/f"{row['index']:02d}_metrics.json",row);print(label,'scored',row['index'],flush=True)
 rows=[read(root/f'{j:02d}_metrics.json') for j in range(16)];result=dict(velocity_sha256=vh,manifest_sha256=sha(OUT/'monitor/manifest.json'),FRC=float(np.mean([r['FRC'] for r in rows])),FRD=float(np.mean([r['FRD'] for r in rows])),S_MSE=float(np.mean([r['S_MSE'] for r in rows])),FRVar=float(np.mean([r['FRVar'] for r in rows])),rows=rows,source_count=16,exact_pairs=1600,scope='fixed full TRAIN control recordings')
 write(summary,result);return result

def main():
 configure_flow();torch.set_num_threads(4);prepare();m=Models();evaluate(m,m.ref,'P2_reference')
if __name__=='__main__':main()
