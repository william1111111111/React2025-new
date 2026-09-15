"""Finite two-arm GPU7 queue, independent logs and serialized exact FRD CPU work."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time
from .prepare import OUT,sha,write

def execute(command,log):
    with open(log,'a') as f:subprocess.run([sys.executable,'-m',*command],stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'7','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4'})

def arm_run(arm):
    status=OUT/(arm+'_queue.json')
    try:
        for step in (1000,3000,6000):
            write(status,dict(stage='training',target=step,updated_unix=time.time()))
            execute(['mode_supervision.train','--arm',arm,'--stop',str(step)],OUT/(arm+'_train.log'))
            write(status,dict(stage='evaluation',checkpoint=step,updated_unix=time.time()))
            execute(['mode_supervision.evaluate','--arm',arm,'--step',str(step)],OUT/(arm+f'_eval{step}.log'))
        write(status,dict(stage='exact_FRD20',updated_unix=time.time()))
        with open(OUT/'frd_cpu.lock','w') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            label=f'seed123_{arm}_step20000'
            execute(['mode_supervision.frd','--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'],OUT/(arm+'_frd.log'))
        if arm=='M1-mode':
            for kind in ('fixed_plan','fixed_noise','oracle'):
                write(status,dict(stage=kind,updated_unix=time.time()))
                execute(['mode_supervision.evaluate','--arm',arm,'--step','6000','--kind',kind],OUT/(arm+'_'+kind+'.log'))
        write(status,dict(stage='complete',updated_unix=time.time()))
    except BaseException as exc:
        write(status,dict(stage='failed',error=repr(exc),updated_unix=time.time()));raise

def main():
    lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'acceptance.json').read_text())['shared_exposure_verified']
    for p,h in json.loads((OUT/'CODE_IDENTITY.json').read_text()).items():assert sha(p)==h
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures=[ex.submit(arm_run,a) for a in ('M0-control','M1-mode')]
        for f in futures:f.result()
    execute(['mode_supervision.report'],OUT/'report.log')
if __name__=='__main__':main()
