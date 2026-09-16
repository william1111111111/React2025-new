"""Detached bounded train/eval coordinator; persistent heartbeat and exit records."""
import concurrent.futures,fcntl,os,subprocess,sys,time,threading
from pathlib import Path
from .common import *
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','NUMBA_NUM_THREADS':'2'}
def run(label,module,args):
    log=OUT/(label+'.log')
    with log.open('a') as f:
        p=subprocess.Popen([sys.executable,'-m',module,*args],env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            write(OUT/(label+'_process.json'),dict(pid=p.pid,stage=module,updated_unix=time.time(),log=str(log)))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    write(OUT/(label+'_exit.json'),dict(returncode=p.returncode,updated_unix=time.time()))
    if p.returncode:raise RuntimeError(f'{label} exited {p.returncode}; see {log}')

def arm(arm):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        train=ex.submit(run,arm+'_train','generator_reward_nft.train',['--arm',arm])
        for step in [1000,4000]:
            p=OUT/'training'/arm/'checkpoints'/f'step_{step:06d}.pt'
            while not p.exists():
                if train.done():train.result();raise RuntimeError(f'{arm} missing {step}')
                time.sleep(15)
            run(arm+f'_DEV80_{step}','generator_reward_nft.evaluate',['--arm',arm,'--step',str(step)])
        run(arm+'_FRD20','generator_reward_nft.frd',['--model',arm+'_step4000','--task',str(OUT/'task_multitarget'/(arm+'_step4000.json')),'--seconds','86400','--max-pairs','2000'])
        train.result()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (OUT/'CALIBRATION.json').exists() or not (OUT/'ACCEPTANCE.json').exists():
        write(OUT/'queue_status.json',dict(stage='waiting_for_preflight',updated_unix=time.time()))
        for marker,pid in read(OUT/'prerequisite_processes.json').items():
            if not (OUT/marker).exists() and not Path(f'/proc/{pid}').exists():raise RuntimeError(f'Preflight process {pid} exited without {marker}; inspect logs')
        time.sleep(15)
    run('freeze','generator_reward_nft.freeze',[])
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fs=[ex.submit(arm,a) for a in ARMS]
            while any(not f.done() for f in fs):
                write(OUT/'queue_status.json',dict(stage='formal_training_and_scheduled_evaluation',updated_unix=time.time()));time.sleep(30)
            for f in fs:f.result()
        run('report','generator_reward_nft.report',[])
        write(OUT/'completion.json',dict(complete=True,updated_unix=time.time()))
    except Exception as e:
        write(OUT/'completion.json',dict(complete=False,error=str(e),updated_unix=time.time()));raise
if __name__=='__main__':main()
