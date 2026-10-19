"""Fresh matched T0/T1 600/1200/2400-step teacher learning curves.

Each GPU owns one serial job queue; no epoch-only resume is used. All six
ensemble points receive full VAL571 all-session FRC/FRDiv and exact FRD.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from coreset_reactor.teacher_cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]
STEPS = (600, 1200, 2400)


def compare_states(current, original):
    new = torch.load(current, map_location='cpu')['state_dict']
    old = torch.load(original, map_location='cpu')['state_dict']
    if new.keys() != old.keys():
        raise ValueError('Frozen/current model state keys differ')
    max_abs, sum_sq, count, identical = 0., 0., 0, True
    for key in new:
        delta = new[key].double()-old[key].double()
        max_abs = max(max_abs, float(delta.abs().max()))
        sum_sq += float(delta.square().sum())
        count += delta.numel()
        identical = identical and torch.equal(new[key], old[key])
    return {'bitwise_identical': identical, 'max_abs_parameter_difference': max_abs,
            'rms_parameter_difference': (sum_sq/count)**.5,
            'frozen_checkpoint_sha256': sha256_file(original),
            'new_checkpoint_sha256': sha256_file(current)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 5, 6])
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Need distinct GPU IDs')
    args.run_root = args.run_root.resolve()
    args.frozen_root = args.frozen_root.resolve()
    args.target_cache = args.target_cache.resolve()
    frozen = json.loads((args.frozen_root/'pilot_provenance.json').read_text())
    if (frozen['max_train_steps'], frozen['batch_size'], frozen['seed'],
        frozen['epochs_formal_schedule']) != (600, 32, 1, 50):
        raise ValueError('Unexpected frozen teacher protocol')
    for name, expected in frozen['faceverse_stats'].items():
        if sha256_file(args.asset_mam_root/'external/FaceVerse'/name) != expected:
            raise ValueError(f'FaceVerse normalization changed: {name}')
    manifests = {}
    for arm in ('T0', 'T1'):
        for rank in range(10):
            label = f'{arm}/rank_{rank:02d}'
            manifest = args.frozen_root/'manifests'/arm/f'rank_{rank:02d}.json'
            checksum = sha256_file(manifest)
            if checksum != frozen['target_manifest_sha256'][label]:
                raise ValueError(f'Frozen target manifest changed: {label}')
            manifests[label] = {'path': str(manifest), 'sha256': checksum}
    args.run_root.mkdir(parents=True, exist_ok=False)
    (args.run_root/'eval').mkdir()
    provenance = {
        'experiment': 'matched T0/T1 teacher convergence; fresh uninterrupted training',
        'base_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'frozen_provenance_sha256': sha256_file(args.frozen_root/'pilot_provenance.json'),
        'frozen_initial_state_sha256': frozen['parameter_initialization_sha256'],
        'source_hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT/'mam_reactor/regnn/train_conditional_regnn.py',
            ROOT/'mam_reactor/regnn/conditional_model.py', ROOT/'mam_reactor/regnn/conditional_data.py',
            ROOT/'coreset_reactor/eval_expert_student_fast.py')},
        'manifest_hashes': {k: v['sha256'] for k, v in manifests.items()},
        'faceverse_stats': frozen['faceverse_stats'],
        'steps': list(STEPS), 'arms': ['T0', 'T1'], 'ranks': list(range(10)),
        'seed': 1, 'batch_size': 32, 'formal_schedule_epochs': 50,
        'actual_updates_per_model': 2400, 'models': 20,
        'total_actual_optimizer_updates': 48000,
        'gpus': args.gpus, 'target_cache': str(args.target_cache),
        'precision': 'bf16', 'alignment': 'relative_time_masked',
        'resume_policy': 'none; original checkpoints retained; 600/1200/2400 are continuous prefixes',
    }
    (args.run_root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    shared_env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
                      PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    jobs = [(arm, rank) for rank in range(10) for arm in ('T0', 'T1')]
    started = time.monotonic()

    def train_queue(gpu, queue):
        env = dict(shared_env, CUDA_VISIBLE_DEVICES=str(gpu))
        completed = []
        for arm, rank in queue:
            run = args.run_root/'arms'/arm/f'rank_{rank:02d}'
            run.mkdir(parents=True, exist_ok=False)
            label = f'{arm}/rank_{rank:02d}'
            command = [sys.executable, '-m', 'regnn.train_conditional_regnn',
                       '--data-dir', str(args.data_root), '--run-dir', str(run),
                       '--epochs', '50', '--batch-size', '32', '--workers', '8',
                       '--precision', 'bf16', '--seed', '1', '--fixed-pair-seed', '1',
                       '--fixed-pair-rank', str(rank), '--fixed-pair-count', '10',
                       '--fixed-pair-manifest', manifests[label]['path'],
                       '--fixed-pair-alignment-policy', 'relative_time_masked',
                       '--graph-layers', '2', '--prediction-head', 'linear',
                       '--max-train-steps', '2400', '--snapshot-steps', '600', '1200', '2400',
                       '--save-every', '5', '--print-every', '100']
            (run/'command.json').write_text(json.dumps(command, indent=2))
            with (run/'train.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            audits = {step: json.loads((run/f'conditional-step{step:06d}-seed1.json').read_text()) for step in STEPS}
            if any(a['initial_state_sha256'] != frozen['parameter_initialization_sha256'] for a in audits.values()):
                raise ValueError(f'Initialization differs from frozen teachers: {label}')
            original = args.frozen_root/'arms'/arm/f'rank_{rank:02d}'/'conditional-epoch0006-seed1.pth'
            parity = compare_states(run/'conditional-step000600-seed1.pth', original)
            (run/'frozen600_parameter_parity.json').write_text(json.dumps(parity, indent=2))
            completed.append(label)
            print(json.dumps({'trained': label, 'gpu': gpu, 'snapshots': list(STEPS),
                              'prefix_parameter_parity': parity}), flush=True)
        return completed

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        tasks = [executor.submit(train_queue, gpu, jobs[index::len(args.gpus)])
                 for index, gpu in enumerate(args.gpus)]
        for task in tasks:
            task.result()
    audits = {}
    for step in STEPS:
        rows = [json.loads((args.run_root/'arms'/arm/f'rank_{rank:02d}'/
                           f'conditional-step{step:06d}-seed1.json').read_text()) for arm, rank in jobs]
        for key in ('initial_state_sha256', 'observed_source_schedule_sha256', 'global_step', 'parameter_count'):
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f'Matched training audit failed: {step}/{key}')
        audits[step] = rows[0]
    provenance.update(matched_training_checks='passed', step_audits=audits,
                      training_wall_seconds=time.monotonic()-started)
    (args.run_root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('All 20 teachers trained; initialization/source schedules matched at every milestone.', flush=True)

    eval_jobs = [(arm, step) for step in STEPS for arm in ('T0', 'T1')]

    def evaluate_queue(gpu, queue, exact):
        env = dict(shared_env, CUDA_VISIBLE_DEVICES=str(gpu))
        for arm, step in queue:
            checkpoints = [args.run_root/'arms'/arm/f'rank_{rank:02d}'/
                           f'conditional-step{step:06d}-seed1.pth' for rank in range(10)]
            # The fast pass publishes complete quality/diversity curves before
            # expensive DTW; the full pass remains mandatory for every point.
            suffix = 'full' if exact else 'fast'
            output = args.run_root/'eval'/f'{arm}_s{step:06d}_val571_{suffix}.json'
            command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
                       '--teacher-checkpoints', *map(str, checkpoints),
                       '--data-root', str(args.data_root), '--asset-mam-root', str(args.asset_mam_root),
                       '--output', str(output), '--device', 'cuda:0', '--eval-seed', '1234',
                       '--target-cache', str(args.target_cache)]
            if exact:
                command += ['--exact-frd', '--metric-workers', '16']
            with output.with_suffix('.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({'evaluated': arm, 'step': step, 'exact': exact,
                              'metrics': json.loads(output.read_text())['metrics']}), flush=True)

    # Get all fast curves first, rather than allowing long DTW to hide later
    # milestones. Each worker queue still owns a single physical GPU.
    for exact in (False, True):
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            tasks = [executor.submit(evaluate_queue, gpu, eval_jobs[index::len(args.gpus)], exact)
                     for index, gpu in enumerate(args.gpus)]
            for task in tasks:
                task.result()
    results = {f'{arm}_{step}': json.loads((args.run_root/'eval'/f'{arm}_s{step:06d}_val571_full.json').read_text())
               for arm, step in eval_jobs}
    baseline = results['T0_600']
    for label, result in results.items():
        for key in ('clip_id', 'target_cache_key'):
            if [r[key] for r in result['rows']] != [r[key] for r in baseline['rows']]:
                raise ValueError(f'Matched evaluation mismatch: {label}/{key}')
    (args.run_root/'summary.json').write_text(json.dumps({
        'matched_training_checks': 'passed', 'matched_evaluation_checks': 'passed',
        'metrics': {k: r['metrics'] for k, r in results.items()},
        'checkpoint_sha256': {k: r['checkpoint_sha256'] for k, r in results.items()},
        'total_wall_seconds': time.monotonic()-started}, indent=2))


if __name__ == '__main__':
    main()
