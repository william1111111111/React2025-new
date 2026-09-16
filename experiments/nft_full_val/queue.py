import os,sys,subprocess,time,concurrent.futures,fcntl
from .common import *
def run(label,module,args=[]):
 with (ROOT/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4'})
  while p.poll() is None:
   write(ROOT/(label+'_process.json'),dict(pid=p.pid,updated_unix=time.time(),stage=module))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
 write(ROOT/(label+'_exit.json'),dict(returncode=p.returncode));assert p.returncode==0,(label,p.returncode)
def main():
 ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run('prepare','experiments.nft_full_val.prepare');run('targets','experiments.nft_full_val.targets')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fs=[ex.submit(run,a,'experiments.nft_full_val.evaluate',['--arm',a]) for a in ARMS]
  for f in fs:f.result()
 write(ROOT/'completion.json',dict(complete=True))
if __name__=='__main__':main()
