"""Same T1 fixed identities, changed TRAIN target representation, 600 steps."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from coreset_reactor.teacher_cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--target-cache', type=Path, required=True)
    args = p.parse_args()
    for name in ('root', 'data_root', 'asset_mam_root', 'frozen_root', 'convergence_root', 'target_cache'):
        setattr(args, name, getattr(args, name).resolve())
    cache = args.root/'processed_T1_targets.pt'
    if not cache.exists():
        raise FileNotFoundError(cache)
    run = args.root/'processed_pilot600'
    run.mkdir(exist_ok=False)
    expected = json.loads((args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.json').read_text())
    metadata = json.loads(cache.with_suffix('.json').read_text())
    if metadata['manifest_sha256'] != sha256_file(args.frozen_root/'manifests/T1/rank_00.json'):
        raise ValueError('Target identities changed')
    names = ('run_processed_teacher_pilot.py', 'train_processed_teacher.py', 'processed_teacher_targets.py',
             'eval_expert_student_fast.py', 'fast_frc.py')
    source_hashes = {f'coreset_reactor/{name}': sha256_file(ROOT/'coreset_reactor'/name) for name in names}
    for name in ('train_conditional_regnn.py', 'conditional_data.py', 'conditional_model.py'):
        source_hashes[f'mam_reactor/regnn/{name}'] = sha256_file(ROOT/'mam_reactor/regnn'/name)
    provenance = {'arm': 'T1P', 'seed': 1, 'steps': 600, 'batch_size': 32,
                  'base_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': source_hashes, 'processed_target_cache_sha256': sha256_file(cache),
                  'cache_metadata': metadata,
                  'target_manifest_sha256': {str(i): sha256_file(args.frozen_root/f'manifests/T1/rank_{i:02d}.json') for i in range(10)},
                  'q0': 'reuse T1_600 paired anchor; same target as baseline',
                  'new_optimizer_updates': 5400, 'ensemble_budget_including_reused_q0': 6000,
                  'training_loss': 'unchanged MSE+(1-CCC)+0.05 masked velocity',
                  'changed_factor': 'q1..q9 GT representation only: official VAE processing instead of relative-time interpolation',
                  'limitations': ['TRAIN cache has fixed GT/length seeds; not identical to per-context eval sampling',
                                  'concurrent diagnostics make throughput a non-isolated measurement'],
                  'predeclared_guard': 'FRC >= T1_600, exact_FRD <= T1_600, FRDiv >= T1_600'}
    (run/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def execute(command, gpu, log):
        with log.open('x') as f:
            subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    def train(gpu, ranks):
        for rank in ranks:
            directory = run/f'rank_{rank:02d}'
            directory.mkdir()
            command = [sys.executable, '-m', 'coreset_reactor.train_processed_teacher',
                       '--processed-target-cache', str(cache), '--data-dir', str(args.data_root),
                       '--run-dir', str(directory), '--epochs', '50', '--batch-size', '32',
                       '--workers', '8', '--precision', 'bf16', '--seed', '1', '--fixed-pair-seed', '1',
                       '--fixed-pair-rank', str(rank), '--fixed-pair-count', '10',
                       '--fixed-pair-manifest', str(args.frozen_root/f'manifests/T1/rank_{rank:02d}.json'),
                       '--fixed-pair-alignment-policy', 'relative_time_masked', '--graph-layers', '2',
                       '--prediction-head', 'linear', '--max-train-steps', '600', '--snapshot-steps', '600',
                       '--save-every', '5', '--print-every', '100']
            (directory/'command.json').write_text(json.dumps(command, indent=2))
            execute(command, gpu, directory/'train.log')
            audit = json.loads((directory/'conditional-step000600-seed1.json').read_text())
            for key in ('initial_state_sha256', 'observed_source_schedule_sha256', 'global_step', 'parameter_count'):
                if audit[key] != expected[key]:
                    raise ValueError(f'Matched audit mismatch: {rank}/{key}')
            print(f'T1P rank {rank}: trained, matched audit passed', flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(train, 5, list(range(1, 10, 2))), pool.submit(train, 6, list(range(2, 10, 2)))]
        for task in tasks:
            task.result()
    provenance['matched_training_checks'] = 'passed'
    provenance['training_wall_seconds'] = time.monotonic()-started
    (run/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    checkpoints = [args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.pth'] + [
        run/f'rank_{i:02d}/conditional-step000600-seed1.pth' for i in range(1, 10)]
    for suffix in ('fast', 'full'):
        output = run/f'val571_{suffix}.json'
        command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
                   '--teacher-checkpoints', *map(str, checkpoints), '--data-root', str(args.data_root),
                   '--asset-mam-root', str(args.asset_mam_root), '--device', 'cuda:0',
                   '--eval-seed', '1234', '--output', str(output), '--target-cache', str(args.target_cache)]
        if suffix == 'full':
            command += ['--exact-frd', '--metric-workers', '16']
        execute(command, 6, output.with_suffix('.log'))
        print(json.dumps({'T1P_eval': suffix, 'metrics': json.loads(output.read_text())['metrics']}), flush=True)
    result = json.loads((run/'val571_full.json').read_text())
    baseline = json.loads((args.convergence_root/'eval/T1_s000600_val571_full.json').read_text())
    for key in ('clip_id', 'target_cache_key'):
        assert [r[key] for r in result['rows']] == [r[key] for r in baseline['rows']]
    a, b = result['metrics'], baseline['metrics']
    summary = {'metrics': a, 'delta_vs_T1_600': {k: a[k]-b[k] for k in ('FRC', 'exact_FRD', 'FRDiv', 'FRVar')},
               'strict_guard_pass': a['FRC'] >= b['FRC'] and a['exact_FRD'] <= b['exact_FRD'] and a['FRDiv'] >= b['FRDiv'],
               'matched_training_checks': 'passed', 'matched_evaluation_checks': 'passed',
               'wall_seconds': time.monotonic()-started}
    (run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
