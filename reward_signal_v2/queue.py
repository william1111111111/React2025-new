import concurrent.futures,os,subprocess,sys,time
from .audit import OUT
from reward_policy.common import write

def worker(arm):
    with (OUT/(arm+'_train.log')).open('a') as f:
        p=subprocess.Popen([sys.executable,'-m','reward_signal_v2.train','--arm',arm],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4'})
        while p.poll() is None:
            write(OUT/(arm+'_queue.json'),dict(stage='training',pid=p.pid,gpu=0,updated_unix=time.time()))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    write(OUT/(arm+'_queue.json'),dict(stage='complete' if p.returncode==0 else 'failed',returncode=p.returncode,updated_unix=time.time()))
    if p.returncode:raise RuntimeError(f'{arm} exited {p.returncode}')

def main():
    import fcntl
    lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fs=[ex.submit(worker,a) for a in ('R0-baselined','R1-calibrated')]
        for f in fs:f.result()
    subprocess.run([sys.executable,'-m','reward_signal_v2.report'],check=True)
if __name__=='__main__':main()
