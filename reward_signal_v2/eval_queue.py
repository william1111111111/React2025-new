"""User-requested joint evaluation; separate output ownership per arm."""
import concurrent.futures,fcntl,os,subprocess,sys,time
from .audit import OUT
from reward_policy.common import write

def worker(arm):
    for stage,module,args in [
        ('DEV80','reward_signal_v2.evaluate',['--arm',arm]),
        ('exact_FRD20','reward_signal_v2.frd',['--model',arm+'_step100','--task',str(OUT/'task_multitarget'/(arm+'_step100.json')),'--seconds','86400','--max-pairs','2000'])]:
        log=OUT/(arm+'_'+stage+'.log')
        with log.open('a') as f:
            p=subprocess.Popen([sys.executable,'-m',module,*args],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','NUMBA_NUM_THREADS':'2'})
            while p.poll() is None:
                write(OUT/(arm+'_eval_status.json'),dict(stage=stage,pid=p.pid,updated_unix=time.time(),log=str(log)))
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        write(OUT/(arm+'_'+stage+'_exit.json'),dict(returncode=p.returncode,updated_unix=time.time()))
        if p.returncode:
            write(OUT/(arm+'_eval_status.json'),dict(stage='failed',failed_stage=stage,returncode=p.returncode,log=str(log)))
            raise RuntimeError(f'{arm} {stage}: {p.returncode}')
    write(OUT/(arm+'_eval_status.json'),dict(stage='complete',updated_unix=time.time()))

def main():
    lock=open(OUT/'eval_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fs=[ex.submit(worker,a) for a in ('R0-baselined','R1-calibrated')]
        for f in fs:f.result()
    write(OUT/'eval_completion.json',dict(complete=True,updated_unix=time.time()))
if __name__=='__main__':main()
