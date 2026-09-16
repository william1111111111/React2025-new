"""Bounded two-arm run; stage heartbeat, no automatic budget extension."""
import os,sys,time,subprocess,concurrent.futures,fcntl
from .common import *
def run(label,module,args=[]):
 with (OUT/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','NUMBA_NUM_THREADS':'1'})
  while p.poll() is None:
   write(OUT/(label+'_process.json'),dict(pid=p.pid,updated_unix=time.time(),stage=module))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
 write(OUT/(label+'_exit.json'),dict(returncode=p.returncode));assert p.returncode==0,(label,p.returncode)
def arm(a):
 run(a+'_train','generator_quality_guard.train',['--arm',a]);completion=read(OUT/'training'/a/'completion.json')
 for v in ['last-proposal','last-accepted']:
  label=a+'_'+v;run(label+'_DEV','generator_quality_guard.evaluate',['--arm',a,'--variant',v])
  if v=='last-accepted' and completion['accepted_steps']==0:
   p=Path('runs/reaction_flow/bert_semantic_v1/frd20/seed123_P2-bert_step15000/status_000.json');ref=read(p);assert ref['completed_pairs']==2000
   write(OUT/'frd20'/label/'status_000.json',dict(**{k:x for k,x in ref.items() if k not in ['model','new_pairs']},model=label,new_pairs=0,reused_pairs=2000,source_status=str(p),source_sha256=sha(p),scope='accepted velocity is exactly P2; DEV exports reused and verified'))
  else:run(label+'_FRD','generator_quality_guard.frd',['--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'])
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while not (OUT/'ACCEPTANCE.json').exists() or not (OUT/'monitor/P2_reference/summary.json').exists():
   for marker,proc in [('ACCEPTANCE.json','acceptance_process.json'),('monitor/P2_reference/summary.json','monitor_prepare_process.json')]:
    if not (OUT/marker).exists() and not Path('/proc/'+str(read(OUT/proc)['pid'])).exists():raise RuntimeError('preflight failed: '+marker)
   write(OUT/'queue_status.json',dict(stage='preflight',updated_unix=time.time()));time.sleep(15)
  run('freeze','generator_quality_guard.freeze')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
   fs=[ex.submit(arm,a) for a in ARMS]
   while any(not f.done() for f in fs):write(OUT/'queue_status.json',dict(stage='training_or_final_evaluation',updated_unix=time.time()));time.sleep(30)
   for f in fs:f.result()
  run('report','generator_quality_guard.report');write(OUT/'completion.json',dict(complete=True))
 except Exception as e:write(OUT/'completion.json',dict(complete=False,error=str(e)));raise
if __name__=='__main__':main()
