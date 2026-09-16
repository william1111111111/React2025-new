"""Full native exact FRD, all TEST sources x K10 x 10 targets; no FRD20 shortcut."""
import argparse,concurrent.futures,multiprocessing,time
import numpy as np
import torch
from tslearn.metrics import dtw
from .common import *
from hirp.phase24 import verify_files

def source(args):
 arm,i,export,target,identity=args;torch.set_num_threads(1);dest=ROOT/'frd'/arm;dest.mkdir(parents=True,exist_ok=True);path=dest/f'{i:04d}.json'
 if path.exists():
  row=read(path);assert row['identity']==identity;matrix=row['matrix']
 else:matrix=[[None]*10 for _ in range(10)]
 verify_files([export,target]);p=np.load(export['path']);y=np.load(target['path']);assert p.shape==y.shape
 for k in range(10):
  for j in range(10):
   if matrix[k][j] is not None:continue
   value=sum(w*float(dtw(p[k,:,lo:hi].astype(np.float32),y[j,:,lo:hi].astype(np.float32))) for lo,hi,w in [(0,15,1/15),(15,17,1),(17,25,1/8)])
   assert np.isfinite(value);matrix[k][j]=value;write(path,dict(identity=identity,matrix=matrix,complete=all(v is not None for row in matrix for v in row)))
 return i,float(np.array(matrix).min(1).sum())

def main(arm):
 result=read(ROOT/(arm+'_RESULTS.json'));target=read(ROOT/'targets.json');n=len(result['rows']);values={};identity=dict(result_sha256=sha(ROOT/(arm+'_RESULTS.json')),targets_sha256=sha(ROOT/'targets.json'),implementation_sha256=sha(__file__))
 tasks=[(arm,i,row['export'],target['files'][i],identity) for i,row in enumerate(result['rows'])]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(source,t) for t in tasks]
  for f in concurrent.futures.as_completed(fs):
   i,v=f.result();values[i]=v;write(ROOT/(arm+'_FRD_status.json'),dict(completed_sources=len(values),total_sources=n,completed_pairs=len(values)*100,requested_pairs=n*100,updated_unix=time.time()));print(arm,'FRD',len(values),'/',n,flush=True)
 groups={'full_local_TEST':list(values),'original_speaker_direction':[i for i in values if result['rows'][i]['clip_id'].startswith('speaker/')],'reverse_direction':[i for i in values if result['rows'][i]['clip_id'].startswith('listener/')]}
 write(ROOT/(arm+'_FRD_RESULTS.json'),dict(completed=True,pairs=n*100,results={g:dict(count=len(ix),FRD=float(np.mean([values[i] for i in ix]))) for g,ix in groups.items() if ix},per_source=values))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
