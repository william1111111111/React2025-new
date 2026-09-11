"""Finite sequential stages. No automatic budget extension or model selection."""
import argparse,json,subprocess,sys
from pathlib import Path
from .phase15_audit import sha256_file
from .train_phase22 import write

ROOT=Path('runs/phase25/timescale_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--seed',type=int,choices=[123,42,2026],default=123);p.add_argument('--through',type=int,choices=[2000,6000],default=2000);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    run=ROOT/f'seed_{a.seed}'
    logdir=ROOT/'commands';logdir.mkdir(exist_ok=True)
    launch=logdir/f'launch_{len(list(logdir.glob("launch_*"))):03d}';launch.mkdir()
    write(launch/'source_snapshot.json',{str(p):sha256_file(p) for p in sorted(Path('hirp').rglob('*.py'))})
    for arm in ('C0','C1','C2'):
        paths=sorted((run/arm).glob('attempt_*/checkpoints/step_*.pt'))
        checkpoint=max(paths,key=lambda p:int(p.stem.split('_')[1])) if paths else None
        step=int(checkpoint.stem.split('_')[1]) if checkpoint else 0
        if step<a.through:
            cmd=[sys.executable,'-m','hirp.train_phase25','--run',str(run),'--device',a.device,'--arm',arm,'--init-seed',str(a.seed),'--sampler-seed',str(a.seed),'--noise-seed',str(a.seed),'--stop-after',str(a.through)]
            if checkpoint:cmd+=['--resume',str(checkpoint)]
            write(launch/f'{arm}_command.json',cmd)
            with (launch/f'{arm}.txt').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        if a.evaluate:
            for target in (128,500,1000,2000,4000,6000):
                if target>a.through:continue
                paths=sorted((run/arm).glob(f'attempt_*/checkpoints/step_{target:06d}.pt'))
                if not paths:raise ValueError(f'missing fixed checkpoint {arm}/{target}')
                cmd=[sys.executable,'-m','hirp.task_phase25','--checkpoint',str(paths[0]),'--device',a.device]
                write(launch/f'{arm}_eval_{target}_command.json',cmd)
                with (launch/f'{arm}_eval_{target}.txt').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
