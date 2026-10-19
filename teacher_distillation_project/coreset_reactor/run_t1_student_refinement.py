"""Run three strictly matched five-epoch T1 student continuation arms."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=3, default=[1, 5, 6])
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.parent_run_root = args.parent_run_root.resolve()
    cache = args.parent_run_root/'train_cache/manifest.json'
    initial = args.parent_run_root/'contrast/student.pth'
    if not cache.is_file() or not initial.is_file():
        raise FileNotFoundError('Frozen teacher cache and contrast student are required')
    args.run_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    arms = ('control', 'au_decision', 'au_logit')

    def run_arm(arm, gpu):
        run = args.run_root/arm
        command = [sys.executable, '-m', 'coreset_reactor.refine_expert_student',
                   '--data-root', str(args.data_root), '--cache-manifest', str(cache),
                   '--initialize-student', str(initial), '--run-dir', str(run),
                   '--device', f'cuda:{gpu}', '--objective', arm, '--epochs', '5',
                   '--batch-size', '4', '--workers', '4', '--seed', '1', '--lr', '0.00002',
                   '--extra-grad-ratio', '0.25']
        (args.run_root/f'{arm}_command.json').write_text(json.dumps(command, indent=2))
        with (args.run_root/f'{arm}_train.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{arm}: training complete', flush=True)
        for exact in (False, True):
            output = run/('val571_full.json' if exact else 'val571_fast.json')
            command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
                       '--checkpoint', str(run/'student.pth'), '--data-root', str(args.data_root),
                       '--asset-mam-root', str(args.asset_mam_root), '--output', str(output),
                       '--device', f'cuda:{gpu}', '--eval-seed', '1234',
                       '--target-cache', str(args.target_cache)]
            if exact:
                command += ['--exact-frd', '--metric-workers', '16']
            with output.with_suffix('.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({'arm': arm, 'exact': exact,
                              'metrics': json.loads(output.read_text())['metrics']}), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks = [executor.submit(run_arm, arm, gpu) for arm, gpu in zip(arms, args.gpus)]
        for task in tasks:
            task.result()
    provenance = {a: json.loads((args.run_root/a/'provenance.json').read_text()) for a in arms}
    configs = {a: json.loads((args.run_root/a/'config.json').read_text()) for a in arms}
    for key in ('initialization_checkpoint_hash', 'initial_student_state_sha256',
                'observed_source_schedule_sha256', 'cache_manifest_hash'):
        if len({p[key] for p in provenance.values()}) != 1:
            raise ValueError(f'Matched comparison failed: {key}')
    for key in ('epochs', 'batch_size', 'seed', 'lr', 'updates', 'relation_weight', 'parameter_count'):
        if len({c[key] for c in configs.values()}) != 1:
            raise ValueError(f'Matched training failed: {key}')
    results = {a: json.loads((args.run_root/a/'val571_full.json').read_text()) for a in arms}
    for arm in arms[1:]:
        for key in ('clip_id', 'target_cache_key'):
            if [r[key] for r in results[arm]['rows']] != [r[key] for r in results['control']['rows']]:
                raise ValueError(f'Matched evaluation failed: {arm}/{key}')
    (args.run_root/'matched_summary.json').write_text(json.dumps({
        'matched_checks': 'passed', 'parent_run_root': str(args.parent_run_root),
        'metrics': {a: r['metrics'] for a, r in results.items()}, 'provenance': provenance,
        'configs': configs}, indent=2))


if __name__ == '__main__':
    main()
