import subprocess,sys,time,concurrent.futures,fcntl,os
from reaction_reward.common import ROOT,read,write,sha
from .train import OUT,ARMS

def run(label,module,args=()):
 with (OUT/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'3'})
  while p.poll() is None:
   write(OUT/(label+'_process.json'),dict(pid=p.pid,time=time.time()))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
 write(OUT/(label+'_exit.json'),dict(returncode=p.returncode,time=time.time()));assert p.returncode==0,(label,p.returncode)
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while not all((OUT/f'bank_complete_shard{i}.json').exists() for i in [0,1]):
   for i in [0,1]:
    if not (OUT/f'bank_complete_shard{i}.json').exists():assert __import__('pathlib').Path('/proc/'+str(read(OUT/f'restore_{i}_process.json')['pid'])).exists(),'restoration stopped; inspect log'
   write(OUT/'queue_status.json',dict(stage='restoring_hash_identical_bank',training_started=False,time=time.time()));time.sleep(20)
  assert read(OUT/'SMOKE.json')['passed'];assert read(OUT/'PREPARE_COMPLETE.json')['complete']
  for p,h in read(OUT/'PROTOCOL.json')['input_hashes'].items():assert sha(ROOT/p)==h
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
   fs=[ex.submit(run,arm,'reaction_reward.quality_v2.train',('--arm',arm)) for arm in ARMS]
   while any(not f.done() for f in fs):
    write(OUT/'queue_status.json',dict(stage='A_B_parallel_training',training_started=True,time=time.time()));time.sleep(30)
   for f in fs:f.result()
  run('evaluation','reaction_reward.quality_v2.evaluate');write(OUT/'completion.json',dict(complete=True,RL_started=False,time=time.time()))
 except Exception as e:write(OUT/'completion.json',dict(complete=False,error=str(e),time=time.time()));raise
if __name__=='__main__':main()
