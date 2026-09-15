"""Evaluate completed policies on GPU0; hold only the coordinator handoff."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/reaction_flow/reward_policy_v1'

def write(name,row):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**row),indent=2));t.replace(p)

def main():
    handoff=4090199
    assert b'parallel_arm.py' in Path(f'/proc/{handoff}/cmdline').read_bytes()
    os.kill(handoff,signal.SIGSTOP)
    write('gpu0_eval_amendment.json',dict(user_authorized=True,gpu=0,arms=['R-quality','R-distance'],held_handoff_pid=handoff,coverage_training_unchanged=True,scope='formal DEV80; original queue will perform exact FRD20 after handoff',protocol_metrics_noise_unchanged=True))
    try:
        for arm in ('R-quality','R-distance'):
            log=OUT/(arm+'_evaluate.log')
            with log.open('a') as f:
                p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','reward_policy.evaluate','--arm',arm],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4'})
                start=time.time()
                while p.poll() is None:
                    write(arm+'_queue.json',dict(stage='reward_policy.evaluate',pid=p.pid,gpu=0,elapsed_seconds=time.time()-start,log=str(log)))
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
            write(arm+'_gpu0_exit.json',dict(returncode=p.returncode,signal=signal.Signals(-p.returncode).name if p.returncode<0 else None))
            if p.returncode:raise RuntimeError(f'{arm} evaluation failed; original queue can resume partial exports')
    finally:
        assert b'parallel_arm.py' in Path(f'/proc/{handoff}/cmdline').read_bytes()
        os.kill(handoff,signal.SIGCONT);write('gpu0_handoff.json',dict(resumed_supervisor_pid=handoff))
if __name__=='__main__':main()
