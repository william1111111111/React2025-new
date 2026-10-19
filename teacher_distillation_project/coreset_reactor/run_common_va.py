"""Run frozen proposal generation, deterministic VAL64 pilot, then full571."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from coreset_reactor.teacher_cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'baseline', 'quality', 'target-cache', 'data-root', 'asset-mam-root']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.root.exists():
        raise FileExistsError(args.root)
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=str(args.gpu))
    base = [sys.executable, '-m', 'coreset_reactor.common_va_experiment']

    def execute(command, log):
        with log.open('x') as handle:
            subprocess.run(command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)

    generation = base+['generate', '--root', str(args.root)]
    for name in ['baseline', 'quality', 'target_cache', 'data_root', 'asset_mam_root']:
        generation += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    execute(generation, args.root.with_suffix('.generation.log'))
    print('571 correction proposal families generated; no model trained', flush=True)
    for phase in ['pilot', 'full']:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(execute, base+['score', '--root', str(args.root), '--phase', phase,
                                               '--shard-count', '3', '--shard-index', str(i), '--workers', '10'],
                                args.root/f'{phase}_shard{i}.log') for i in range(3)]
            for i, job in enumerate(jobs):
                job.result()
                print(f'{phase} shard{i}: complete', flush=True)
        name = 'COMMON_VA_PILOT.md' if phase == 'pilot' else 'COMMON_VA.md'
        execute(base+['summarize', '--root', str(args.root), '--phase', phase,
                      '--report', str(ROOT/'coreset_reactor/results'/name)], args.root/f'{phase}_summary.log')
        result = json.loads((args.root/f'{phase}_summary.json').read_text())
        assert result['strict_all_contexts_safe'] and result['geometry_max_error'] < 3e-7
        print(json.dumps({'phase': phase, 'metrics': result['metrics'],
                          'selection': result['selection_composition']}), flush=True)
    provenance = json.loads((args.root/'provenance.json').read_text())
    for name, expected in provenance['source_hashes'].items():
        assert sha256_file(ROOT/name) == expected, name
    print('Completed fixed protocol, all source hashes intact', flush=True)


if __name__ == '__main__':
    main()
