"""Run matched T1 Basic, distance, and signed-contrast KD students."""
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
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 5, 6])
    parser.add_argument('--arms', nargs='+', choices=('basic', 'distance', 'contrast', 'au_decision'),
                        default=['basic', 'distance', 'contrast'])
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    cache = args.run_root / 'train_cache/manifest.json'
    if not cache.is_file():
        raise FileNotFoundError('Generate the frozen T1 teacher cache first')
    init = args.teacher_root / 'arms/T1/rank_00/conditional-epoch0006-seed1.pth'
    arm_options = {'basic': ('basic', 'distance', 0.0),
                   'distance': ('distance', 'distance', .25),
                   'contrast': ('contrast', 'centered_residual', .25),
                   'au_decision': ('au_decision', 'au_decision', .25)}
    arms = [arm_options[name] for name in args.arms]
    if len(arms) != len(args.gpus) or len(set(args.arms)) != len(arms):
        raise ValueError('Specify one GPU per distinct arm')
    env = dict(os.environ)
    env.update(MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT / "mam_reactor"}:{ROOT}',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def run_arm(spec, gpu):
        name, objective, ratio = spec
        run = args.run_root / name
        train = [sys.executable, '-m', 'coreset_reactor.train_expert_student',
                 '--data-root', str(args.data_root), '--cache-manifest', str(cache),
                 '--run-dir', str(run), '--device', f'cuda:{gpu}',
                 '--epochs', '10', '--batch-size', '4', '--workers', '4',
                 '--lr', '0.0001', '--seed', '1', '--expert-embedding-dim', '64',
                 '--initialize-backbone', str(init), '--relation-objective', objective,
                 '--relation-grad-ratio', str(ratio), '--print-every', '100',
                 '--save-every', '5']
        (args.run_root / f'{name}_command.json').write_text(json.dumps(train, indent=2))
        with (args.run_root / f'{name}_train.log').open('x') as log:
            subprocess.run(train, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f'{name}: training complete', flush=True)
        for exact in (False, True):
            output = run / ('val571_full.json' if exact else 'val571_fast.json')
            command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
                       '--checkpoint', str(run / 'student.pth'),
                       '--data-root', str(args.data_root),
                       '--asset-mam-root', str(args.asset_mam_root),
                       '--output', str(output), '--device', f'cuda:{gpu}',
                       '--eval-seed', '1234', '--target-cache', str(args.target_cache)]
            if exact:
                command += ['--exact-frd', '--metric-workers', '16']
            with output.with_suffix('.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            print(json.dumps({'arm': name, 'exact': exact,
                              'metrics': json.loads(output.read_text())['metrics']}), flush=True)
        return name

    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        futures = [pool.submit(run_arm, arm, gpu) for arm, gpu in zip(arms, args.gpus)]
        for future in futures:
            future.result()
    # Include completed earlier arms in the final comparison of a follow-up.
    arms = [arm_options[name] for name in arm_options
            if (args.run_root / name / 'val571_full.json').exists()]
    provenance = [json.loads((args.run_root / name / 'provenance.json').read_text())
                  for name, _, _ in arms]
    for key in ('initial_student_state_sha256', 'observed_source_schedule_sha256',
                'cache_manifest_hash', 'initialization_checkpoint_hash'):
        if len({p[key] for p in provenance}) != 1:
            raise ValueError(f'Matched comparison failed for {key}')
    results = {name: json.loads((args.run_root / name / 'val571_full.json').read_text())
               for name, _, _ in arms}
    reference_arm = arms[0][0]
    for name, _, _ in arms[1:]:
        for key in ('clip_id', 'target_cache_key'):
            if [r[key] for r in results[name]['rows']] != [r[key] for r in results[reference_arm]['rows']]:
                raise ValueError(f'Evaluation mismatch: {name}/{key}')
    (args.run_root / 'matched_summary.json').write_text(json.dumps({
        'matched_checks': 'passed',
        'metrics': {name: result['metrics'] for name, result in results.items()},
        'provenance': dict(zip((a[0] for a in arms), provenance)),
    }, indent=2))


if __name__ == '__main__':
    main()
