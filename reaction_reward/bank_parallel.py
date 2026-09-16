"""Two disjoint window shards; completion only after every fixed bank is scored."""
import subprocess,sys,time,fcntl
from .common import *
def main():
 lock=(OUT/'bank_parallel.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 children=[];logs=[]
 try:
  for shard,gpu in enumerate([0,1]):
   f=(OUT/f'bank_shard{shard}.log').open('a');logs.append(f);p=subprocess.Popen([sys.executable,'-m','reaction_reward.generated_bank','--shard',str(shard),'--shards','2'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'3','NUMBA_NUM_THREADS':'1'});children.append(p);write(OUT/f'bank_shard{shard}_process.json',dict(pid=p.pid,gpu=gpu,time=time.time()))
  while any(p.poll() is None for p in children):
   for p in children:
    if p.poll() not in [None,0]:raise RuntimeError(f'worker {p.pid} exited {p.returncode}')
   generated=len(list((OUT/'bank').glob('*/*.npy')))*16;scored=len(list((OUT/'bank').glob('*/*_scores.json')))*16
   write(OUT/'bank_status.json',dict(stage='parallel_fixed_candidate_bank',candidates=generated,scored_candidates=scored,max_candidates=14848,workers=2,gpus=[0,1],time=time.time()));time.sleep(10)
  assert all(p.returncode==0 for p in children)
  total=0
  for w in read(OUT/'WINDOWS.json'):
   for k in ['P2','N0']+(['N1'] if w['split']=='RM_audit' else []):
    root=OUT/'bank'/w['id'];h=sha(root/(k+'.npy'));assert read(root/(k+'.json'))['sha256']==h;assert read(root/(k+'_scores.json'))['prediction_sha256']==h;total+=16
  write(OUT/'bank_complete.json',dict(complete=True,candidates=total,time=time.time(),parallel_workers=2))
 except Exception as e:
  for p in children:
   if p.poll() is None:p.terminate()
  write(OUT/'bank_parallel_failure.json',dict(error=str(e),time=time.time()));raise
 finally:
  for f in logs:f.close()
if __name__=='__main__':main()
