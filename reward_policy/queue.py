"""Finite serial GPU7 policy queue with exit status and persistent heartbeat."""
import fcntl,json,os,signal,subprocess,sys,time
from .common import OUT,ARMS,read,write,sha

def event(**row):
    with (OUT/'lifecycle.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**row))+'\n');f.flush();os.fsync(f.fileno())

def execute(arm,module,args):
    command=[sys.executable,'-m',module,*args];log=OUT/(arm+'_'+module.split('.')[-1]+'.log');begin=time.time()
    with log.open('a') as f:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        event(event='child_start',arm=arm,pid=p.pid,command=command,log=str(log))
        while p.poll() is None:
            write(OUT/(arm+'_queue.json'),dict(stage=module,pid=p.pid,updated_unix=time.time(),elapsed_seconds=time.time()-begin,log=str(log),log_age_seconds=time.time()-log.stat().st_mtime))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        code=p.returncode;event(event='child_exit',arm=arm,pid=p.pid,returncode=code,signal=signal.Signals(-code).name if code<0 else None,seconds=time.time()-begin)
        if code:write(OUT/(arm+'_queue.json'),dict(stage='failed',returncode=code,log=str(log)));raise subprocess.CalledProcessError(code,command)

def main():
    lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(OUT/'REACHABILITY.json')['gate_passed'] and read(OUT/'TRAIN_ACCEPTANCE.json')['passed']
    for p,h in read(OUT/'CODE_IDENTITY.json').items():assert sha(p)==h
    for p,h in read(OUT/'PROTOCOL.json')['frozen_inputs'].items():assert sha(p)==h
    event(event='queue_start',pid=os.getpid(),arms=ARMS,policy='serial on GPU7, no automatic budget extension')
    def stop(sig,frame):event(event='queue_signal',signal=signal.Signals(sig).name);raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    # Finish policy training for all arms before the expensive CPU full-record FRD.
    for arm in ARMS:execute(arm,'reward_policy.train',['--arm',arm])
    for arm in ARMS:
        execute(arm,'reward_policy.evaluate',['--arm',arm]);label=arm+'_step500'
        execute(arm,'reward_policy.frd',['--model',label,'--task',str(OUT/'task_multitarget'/(label+'.json')),'--seconds','86400','--max-pairs','2000'])
        write(OUT/(arm+'_queue.json'),dict(stage='complete',updated_unix=time.time()))
    execute('summary','reward_policy.report',[]);event(event='queue_complete')
if __name__=='__main__':main()
