"""Run the frozen TRAIN-only corrector fit and full VAL571 evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','data-root','asset-mam-root','baseline','target-cache','oracle-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--labels', type=Path, nargs=3, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    args.labels = [x.resolve() for x in args.labels]
    if args.root.exists(): raise FileExistsError(args.root)
    args.root.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def execute(command, log):
        with log.open('x') as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)

    fit = args.root/'fit'
    train = [sys.executable, '-m', 'coreset_reactor.train_learned_common_va',
             '--output', str(fit), '--labels', *map(str,args.labels),
             '--data-root', str(args.data_root), '--asset-mam-root', str(args.asset_mam_root),
             '--device', 'cuda:0', '--epochs', '20', '--batch-size', '8', '--workers', '4']
    execute(train, args.root/'train.log')
    checkpoint = fit/'common-va-corrector-epoch0020-seed20260925.pth'
    evaluation = args.root/'evaluation'
    generate = [sys.executable, '-m', 'coreset_reactor.learned_common_va_experiment', 'generate',
                '--root', str(evaluation), '--checkpoint', str(checkpoint),
                '--baseline', str(args.baseline), '--target-cache', str(args.target_cache),
                '--data-root', str(args.data_root), '--asset-mam-root', str(args.asset_mam_root),
                '--device', 'cuda:0']
    execute(generate, args.root/'generate.log')
    base = [sys.executable, '-m', 'coreset_reactor.learned_common_va_experiment', 'score',
            '--root', str(evaluation), '--shard-count', '3', '--workers', '10']
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(execute, base+['--shard-index',str(i)], args.root/f'score_shard{i}.log')
                for i in range(3)]
        for job in jobs: job.result()
    report = ROOT/'coreset_reactor/results/LEARNED_COMMON_VA.md'
    summarize = [sys.executable, '-m', 'coreset_reactor.learned_common_va_experiment', 'summarize',
                 '--root', str(evaluation), '--oracle-root', str(args.oracle_root),
                 '--report', str(report)]
    execute(summarize, args.root/'summarize.log')
    print(f'Complete: {report}', flush=True)


if __name__ == '__main__': main()
