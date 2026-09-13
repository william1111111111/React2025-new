"""Evaluate fixed 500/1000-step checkpoints without restarting training."""
import json, os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def write(value):
    tmp = ROOT / 'intermediate_monitor.tmp'
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(ROOT / 'intermediate_monitor.json')

def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    pending = [(arm, step) for step in (14500, 15000) for arm in ('G0-local', 'G1-shared')]
    completed, failures = [], []
    while pending:
        for arm, step in pending[:]:
            cp = ROOT / arm / 'attempt_000' / 'checkpoints' / f'step_{step:06d}.pt'
            if not cp.exists() or not cp.with_suffix('.json').exists():
                continue
            command = ['.venv/bin/python', '-m', 'reaction_flow.evaluate_shared', '--checkpoint', str(cp)]
            with (ROOT / f'eval_intermediate_{arm}_{step}.txt').open('x') as log:
                proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                write(dict(time=time.time(), active=[arm, step], pid=proc.pid, command=command, completed=completed, failures=failures))
                code = proc.wait()
            (completed if code == 0 else failures).append(dict(arm=arm, step=step, exit_code=code))
            pending.remove((arm, step))
        if (ROOT / 'queue_finished.json').exists() and pending:
            failures.extend(dict(arm=a, step=s, error='checkpoint unavailable after queue ended') for a, s in pending)
            pending.clear()
        write(dict(time=time.time(), active=None, pending=pending, completed=completed, failures=failures))
        if pending:
            time.sleep(30)

if __name__ == '__main__':
    main()
