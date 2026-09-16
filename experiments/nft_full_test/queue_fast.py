"""Resume cached outputs, overlap batched generation / CPU metrics / exact FRD."""
import concurrent.futures,fcntl,subprocess,sys,os,time
from .common import *
from .queue import run as original_run

def run(label,module,args=[]):
 gpu='1' if label=='N1-quality-coverage' else '0'
 with (ROOT/(label+'.log')).open('a') as log:
  p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','NUMBA_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  while p.poll() is None:
   write(ROOT/(label+'_process.json'),dict(pid=p.pid,gpu=gpu,stage=module,updated_unix=time.time()))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
 write(ROOT/(label+'_exit.json'),dict(returncode=p.returncode));assert p.returncode==0,(label,p.returncode)

def arm(a):
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  fs=[ex.submit(run,a,'experiments.nft_full_test.evaluate_fast',['--arm',a]),ex.submit(run,a+'_FRD','experiments.nft_full_test.frd_stream',['--arm',a])]
  for f in fs:f.result()
def main():
 lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (ROOT/'targets.json').exists()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
   fs=[ex.submit(arm,a) for a in ARMS]
   for f in fs:f.result()
  run('report_fast','experiments.nft_full_test.report');write(ROOT/'completion.json',dict(complete=True,execution='batched asynchronous streaming'))
 except Exception as e:write(ROOT/'completion.json',dict(complete=False,error=str(e)));raise
if __name__=='__main__':main()
