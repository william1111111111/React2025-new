"""Fixed TRAIN candidate bank; generator identities never enter the judge."""
import time,concurrent.futures,argparse
import numpy as np
import torch
from dataclasses import replace
from reaction_reward.common import *
NEW=ROOT/'runs/reaction_reward/quality_v2'
from reaction_reward.data import load
from generator_reward_nft.public import Generator
from reward_policy.common import PARENT
from reaction_flow.export import recording_noise
from reaction_flow.train import configure_flow

def score_job(job):
 from tslearn.metrics import dtw
 from generator_quality_guard.quality import ccc
 w,path,targets=job;p=np.load(path);ys=[load(t['path'])[t['start']:t['start']+w['n']] for t in targets];C=[];D=[]
 for v in p:
  C.append([float(ccc(torch.from_numpy(v),torch.from_numpy(y))) for y in ys]);D.append([sum(weight*float(dtw(v[:,a:b],y[:,a:b])) for a,b,weight in [(0,15,1/15),(15,17,1),(17,25,1/8)]) for y in ys])
 return dict(C=C,D=D,prediction_sha256=sha(path),targets=targets)

def main(shard=0,shards=1):
 configure_flow();torch.set_num_threads(3);torch.manual_seed(123)
 windows=read(OUT/'WINDOWS.json');rows={r['id']:r for r in read(OUT/'SPLIT_MANIFEST.json')['records']};gen=Generator();initial={k:v.cpu().clone() for k,v in gen.model.base.velocity.state_dict().items()};parent=ROOT/PARENT
 paths={'P2':parent,'N0':ROOT/'runs/reaction_flow/generator_reward_nft_v1/training/N0-quality/checkpoints/step_001000.pt','N1':ROOT/'runs/reaction_flow/generator_reward_nft_v1/training/N1-quality-coverage/checkpoints/step_001000.pt'}
 for p in paths.values():assert p.exists(),p
 provenance=dict(checkpoints={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},condition='speaker numeric streams only; NULL text for all',noise_seed=9610123,Euler=16,K=16,fit_cal=['P2','N0'],audit=['P2','N0','N1'],generator_audit_training_overlap='historical generators may have seen all TRAIN; only judge is group held out')
 if (OUT/'GENERATOR_PROVENANCE.json').exists():assert read(OUT/'GENERATOR_PROVENANCE.json')==provenance
 else:write(OUT/'GENERATOR_PROVENANCE.json',provenance)
 norm_mean=np.load(ROOT/'external/FaceVerse/mean_face.npy');norm_std=np.maximum(np.load(ROOT/'external/FaceVerse/std_face.npy'),1e-8)
 # Serialize GPU generation; bounded native-DTW CPU workers work concurrently.
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  pending=[];count=0
  for wi,w in enumerate(windows):
   if wi%shards!=shard:continue
   names=['P2','N0']+(['N1'] if w['split']=='RM_audit' else [])
   root=OUT/'bank'/w['id']
   if all((root/(k+'_scores.json')).exists() and (root/(k+'.json')).exists() and (root/(k+'.npy')).exists() for k in names):
    for k in names:
     assert read(root/(k+'.json'))['sha256']==sha(root/(k+'.npy'))
     assert read(root/(k+'_scores.json'))['prediction_sha256']==sha(root/(k+'.npy'))
    count+=16*len(names);continue
   r=rows[w['source']];s=w['start'];n=w['n'];pts=r['pts']['speaker'];streams=[torch.from_numpy(np.array(load(r['files'][f+'/speaker']['path']),copy=True)).float() for f in ['audio-features','facial-attributes','coefficients']];streams[2]=(streams[2]-torch.from_numpy(norm_mean))/torch.from_numpy(norm_std)
   hs,ms,_=gen.features(streams,pts,[None]*((len(pts)+749)//750));noise=recording_noise(replace(gen.model.base.config,evaluation_seed=9610123),w['id'],len(pts),16).cuda().double()
   for model in ['P2','N0']+(['N1'] if w['split']=='RM_audit' else []):
    root=OUT/'bank'/w['id'];root.mkdir(parents=True,exist_ok=True);path=root/(model+'.npy');meta=root/(model+'.json');scores=root/(model+'_scores.json')
    if path.exists():assert read(meta)['sha256']==sha(path)
    else:
     if model=='P2':gen.model.base.velocity.load_state_dict(initial)
     else:gen.model.base.velocity.load_state_dict(torch.load(paths[model],map_location='cpu',weights_only=True)['student'])
     with torch.no_grad():pred=gen.block(hs[s//750],ms[s//750],noise,s,n).float().cpu().numpy()
     assert pred.shape==(16,n,25) and np.isfinite(pred).all();tmp=path.with_suffix('.tmp')
     with tmp.open('wb') as f:np.save(f,pred)
     assert sha(tmp)==read(meta)['sha256'], f'REGENERATION HASH MISMATCH {path}';tmp.replace(path)
    assert scores.exists(), 'Original frozen score matrix is missing'
    if not scores.exists():
     targets=[dict(path=rows[t['id']]['files']['facial-attributes/listener']['path'],start=t['start'],source_id=t['id'],split=w['split']) for t in w['targets']];pending.append((pool.submit(score_job,(w,str(path),targets)),scores))
    count+=16;write(NEW/(f'bank_status_shard{shard}.json' if shards>1 else 'bank_status.json'),dict(stage='fixed_candidate_generation',candidates=count,max_candidates=14848,window=w['id'],model=model,time=time.time()));print('BANK',count,w['id'],model,flush=True)
    # Bound queued memory; score input only stores paths, not model tensors.
    if len(pending)>=6:
     f,p=pending.pop(0);write(p,f.result())
   del hs,ms,noise
  for f,p in pending:write(p,f.result())
 write(NEW/(f'bank_complete_shard{shard}.json' if shards>1 else 'bank_complete.json'),dict(complete=True,candidates=count,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();assert 0<=a.shard<a.shards;main(a.shard,a.shards)
