"""Concurrent formal evaluations with one owner per result directory."""
import concurrent.futures,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/reward_policy_v1';HELD=8327

def write(name,row):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**row),indent=2));t.replace(p)

def run(arm,module,args,gpu):
    log=OUT/(arm+'_'+module.split('.')[-1]+'_parallel.log');begin=time.time()
    with log.open('a') as f:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m',module,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','NUMBA_NUM_THREADS':'2'})
        while p.poll() is None:
            write(arm+'_queue.json',dict(stage=module,pid=p.pid,gpu=gpu if module.endswith('evaluate') else None,elapsed_seconds=time.time()-begin,log=str(log)))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    write(arm+'_'+module.split('.')[-1]+'_parallel_exit.json',dict(returncode=p.returncode,pid=p.pid,signal=signal.Signals(-p.returncode).name if p.returncode<0 else None))
    if p.returncode:raise RuntimeError(f'{arm} {module} exited {p.returncode}')

def arm_run(arm):
    if arm=='R-coverage':run(arm,'reward_policy.evaluate',['--arm',arm],1)
    else:
        # R-quality has finished; R-distance is already owned by the existing worker.
        path=OUT/'task_multitarget'/(arm+'_step500.json')
        while not path.exists():
            write(arm+'_parallel_wait.json',dict(stage='waiting_for_existing_DEV80_worker'));time.sleep(10)
    label=arm+'_step500'
    run(arm,'reward_policy.frd',['--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'],0)
    write(arm+'_queue.json',dict(stage='complete'))

def main():
    assert b'eval_ready.py' in Path(f'/proc/{HELD}/cmdline').read_bytes();os.kill(HELD,signal.SIGSTOP)
    write('parallel_eval_amendment.json',dict(user_authorized=True,held_coordinator=HELD,DEV80={'R-distance':'existing GPU0 worker','R-coverage':'GPU1'},exact_FRD='one CPU process per arm concurrently after its DEV80 result; disjoint pair files',protocol_unchanged=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fs=[ex.submit(arm_run,a) for a in ('R-quality','R-distance','R-coverage')]
        for f in fs:f.result()
    run('summary','reward_policy.report',[],0)
    # All cached results now exist. Original coordinators can finish through their cache checks.
    os.kill(HELD,signal.SIGCONT);write('parallel_eval_complete.json',dict(complete=True))
if __name__=='__main__':main()
