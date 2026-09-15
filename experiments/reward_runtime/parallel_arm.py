"""User-authorized GPU1 coverage run; hold queue coordinator, not its GPU7 worker."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/reward_policy_v1'

def write(name,row):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**row),indent=2));t.replace(p)

def main():
    parent=int(sys.argv[1]);assert b'reward_policy.queue' in Path(f'/proc/{parent}/cmdline').read_bytes()
    # Only the coordinator is paused; R-distance remains running on GPU7.
    os.kill(parent,signal.SIGSTOP)
    write('gpu1_amendment.json',dict(user_authorized=True,queue_pid=parent,training_gpu={'R-distance':7,'R-coverage':1},protocol_hyperparameters_unchanged=True,coordination='pause parent until coverage exits; original queue then loads completed coverage with zero extra updates'))
    with (OUT/'R-coverage_train.log').open('a') as f:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','reward_policy.train','--arm','R-coverage'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4'})
        start=time.time()
        while p.poll() is None:
            write('R-coverage_queue.json',dict(stage='reward_policy.train',pid=p.pid,gpu=1,elapsed_seconds=time.time()-start))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    write('gpu1_exit.json',dict(pid=p.pid,returncode=p.returncode,signal=signal.Signals(-p.returncode).name if p.returncode<0 else None))
    if p.returncode:raise RuntimeError('coverage failed; preserve coordinator pause for inspection')
    assert json.loads((OUT/'training/R-coverage/completion.json').read_text())['training_complete']
    assert b'reward_policy.queue' in Path(f'/proc/{parent}/cmdline').read_bytes()
    os.kill(parent,signal.SIGCONT);write('gpu1_handoff.json',dict(resumed_queue_pid=parent,coverage_training_complete=True))
if __name__=='__main__':main()
