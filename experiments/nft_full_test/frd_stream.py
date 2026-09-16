"""Overlap full exact FRD with generation; immutable per-source exports."""
import argparse,concurrent.futures,multiprocessing,time
from .common import *
from .frd import source

def main(arm):
 m=read(ROOT/'manifest.json');targets=read(ROOT/'targets.json');n=m['population'];values={};submitted=set();pending={};identity=dict(checkpoint_sha256=m['checkpoints'][arm]['sha256'],manifest_sha256=sha(ROOT/'manifest.json'),targets_sha256=sha(ROOT/'targets.json'),implementation_sha256=sha(Path(__file__).with_name('frd.py')))
 with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
  while len(values)<n:
   for i in range(n):
    if len(pending)>=9:break
    if i in submitted:continue
    side=ROOT/'exports'/arm/f'{i:04d}.json'
    if not side.exists():continue
    row=read(side);assert row['checkpoint_sha256']==identity['checkpoint_sha256'];f=pool.submit(source,(arm,i,row['export'],targets['files'][i],identity));pending[f]=i;submitted.add(i)
   if pending:
    done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
    for f in done:
     i,v=f.result();values[i]=v;pending.pop(f);print(arm,'FRD',len(values),'/',n,flush=True)
   else:time.sleep(2)
   write(ROOT/(arm+'_FRD_status.json'),dict(completed_sources=len(values),total_sources=n,completed_pairs=len(values)*100,requested_pairs=n*100,pending_sources=len(pending),stage='streaming_exact_FRD',updated_unix=time.time()))
   exitfile=ROOT/(arm+'_exit.json')
   if exitfile.exists() and read(exitfile)['returncode']!=0:raise RuntimeError('generation failed; partial FRD retained')
 groups={'full_local_TEST':list(values),'original_speaker_direction':[i for i in values if m['sources'][i]['clip_id'].startswith('speaker/')],'reverse_direction':[i for i in values if m['sources'][i]['clip_id'].startswith('listener/')]}
 import numpy as np
 write(ROOT/(arm+'_FRD_RESULTS.json'),dict(completed=True,pairs=n*100,results={g:dict(count=len(ix),FRD=float(np.mean([values[i] for i in ix]))) for g,ix in groups.items() if ix},per_source=values))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);main(p.parse_args().arm)
