"""Stage A only: durable bounded pipeline, never starts generator training."""
import subprocess,time,concurrent.futures,fcntl,sys
from .common import *
from .train import ARMS

def run(label,module,args=()):
 with (OUT/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'3','NUMBA_NUM_THREADS':'1'})
  while p.poll() is None:
   write(OUT/(label+'_process.json'),dict(pid=p.pid,module=module,time=time.time()))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
 write(OUT/(label+'_exit.json'),dict(returncode=p.returncode,time=time.time()));assert p.returncode==0,(label,p.returncode)
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while not (OUT/'bank_complete.json').exists():
   pid=read(OUT/'bank_process.json')['pid'];assert Path('/proc/'+str(pid)).exists(),'candidate bank process exited before completion'
   write(OUT/'queue_status.json',dict(stage='fixed_candidate_bank',time=time.time(),training_started=False));time.sleep(20)
  run('generated_evidence','reaction_reward.generated_evidence')
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
   futures=[ex.submit(run,a,'reaction_reward.train',('--arm',a)) for a in ARMS]
   while any(not f.done() for f in futures):
    write(OUT/'queue_status.json',dict(stage='three_RM_arms',time=time.time(),training_started=True));time.sleep(30)
   for f in futures:f.result()
  for label in ['audit','probes','diagnostics','report']:
   write(OUT/'queue_status.json',dict(stage=label,time=time.time()));run(label,'reaction_reward.'+label)
  write(OUT/'completion.json',dict(complete=True,stage='A',RL_started=False,time=time.time()))
 except Exception as e:write(OUT/'completion.json',dict(complete=False,error=str(e),time=time.time()));raise
if __name__=='__main__':main()
