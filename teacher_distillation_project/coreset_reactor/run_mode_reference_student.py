"""Distill the frozen mode-reference bank using the retained matched KD recipes.

Only TRAIN teacher predictions enter optimization. Full all-session VAL evaluation
is queued after training; this is not the challenge's sampled-ten GT protocol.
Every invocation requires a fresh run directory and retains all old artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VIEW_KEYS = ('sample_index', 'sample_id', 'view_id', 'view_epoch', 'start', 'length',
             'generated_length')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def view_rows(manifest_path):
    import torch
    manifest = read(manifest_path)
    for shard in manifest['shards']:
        for row in torch.load(manifest_path.parent / shard, map_location='cpu'):
            yield tuple(row[key] for key in VIEW_KEYS)


def validate_cache_views(new_path, old_path):
    new, old = read(new_path), read(old_path)
    for key in ('split', 'contexts', 'num_experts', 'view_epoch', 'source_seed',
                'clip_length', 'sampler_config_hash', 'preprocessing_version', 'model_config'):
        if new[key] != old[key]:
            raise ValueError(f'Cache protocol mismatch: {key}')
    if new['split'] != 'train' or new['contexts'] != 3320 or new['num_experts'] != 10:
        raise ValueError('Expected all 3320 TRAIN views and ten canonical teachers')
    new_rows, old_rows = list(view_rows(new_path)), list(view_rows(old_path))
    if new_rows != old_rows or len(new_rows) != new['contexts']:
        raise ValueError('Teacher cache source/crop views differ from retained baseline')
    return {'matched_views': len(new_rows), 'view_sha256': hashlib.sha256(
        json.dumps(new_rows, separators=(',', ':')).encode()).hexdigest()}


def validate_result_alignment(new, old):
    for key in ('clip_id', 'target_cache_key'):
        if [r[key] for r in new['rows']] != [r[key] for r in old['rows']]:
            raise ValueError(f'Full-VAL evaluation mismatch: {key}')
    if len(new['rows']) != 571:
        raise ValueError('Expected full VAL571')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-run', type=Path, required=True)
    parser.add_argument('--paired-checkpoint', type=Path, required=True)
    parser.add_argument('--old-student-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=2, default=[1, 5])
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        raise ValueError('Use two distinct GPUs for matched parallel arms')
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=False)
    status = args.run_root / 'status.json'
    started = time.time()
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT / "mam_reactor"}:{ROOT}',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')

    def stage(name, **extra):
        write(status, {'stage': name, 'pid': os.getpid(), 'started_unix': started,
                       'updated_unix': time.time(), **extra})
        print(name, flush=True)

    def run(name, command):
        write(args.run_root / f'{name}_command.json', command)
        with (args.run_root / f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        stage('preflight')
        old_config = read(args.old_student_run / 'contrast/config.json')
        initial = Path(old_config['initialize_backbone'])
        teacher_root = initial.parents[3]
        expected_init = teacher_root / 'arms/T1/rank_00/conditional-epoch0006-seed1.pth'
        if initial != expected_init:
            raise ValueError('Retained runner initialization path is not the old student initializer')
        teachers = [args.paired_checkpoint] + [args.teacher_run /
                    f'rank_{rank:02d}/conditional-step000600-seed1.pth' for rank in range(1, 10)]
        teacher_eval = read(args.teacher_run / 'val571_full.json')
        if teacher_eval['metrics']['contexts'] != 571:
            raise ValueError('Frozen teacher must have full VAL571 evaluation')
        hashes = [file_hash(p) for p in teachers]
        initial_hash = file_hash(initial)
        for arm in ('basic', 'contrast'):
            if read(args.old_student_run / arm / 'provenance.json')[
                    'initialization_checkpoint_hash'] != initial_hash:
                raise ValueError('Old student initializer hash mismatch')
        files = [Path(__file__).relative_to(ROOT),
                 Path('coreset_reactor/run_t1_student_distillation.py'),
                 Path('coreset_reactor/teacher_cache.py'),
                 Path('coreset_reactor/train_expert_student.py'),
                 Path('coreset_reactor/relation_distillation.py'),
                 Path('coreset_reactor/eval_expert_student_fast.py'),
                 Path('coreset_reactor/fast_frc.py'),
                 Path('coreset_reactor/oracle_select.py'),
                 Path('coreset_reactor/evaluate.py'),
                 Path('mam_reactor/regnn/expert_student.py'),
                 Path('mam_reactor/regnn/conditional_model.py'),
                 Path('mam_reactor/regnn/conditional_data.py')]
        for folder in ('mam_reactor/framework/metrics',):
            files.extend(p.relative_to(ROOT) for p in (ROOT / folder).rglob('*.py'))
        source_hashes = {}
        for relative in files:
            dest = args.run_root / 'source_snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
            source_hashes[str(relative)] = file_hash(dest)
        write(args.run_root / 'experiment.json', {
            'git_commit_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                       text=True).strip(),
            'source_hashes': source_hashes, 'teachers': teachers,
            'teacher_sha256': hashes, 'initialization': initial,
            'initialization_sha256': initial_hash, 'old_student_run': args.old_student_run,
            'teacher_run': args.teacher_run, 'teacher_metrics': teacher_eval['metrics'],
            'arms': ['basic', 'contrast'], 'epochs': 10, 'expected_updates': 8300,
            'seed': 1, 'batch_size': 4, 'gpus': args.gpus,
            'selection': 'Both frozen recipes, no VAL tuning; TRAIN-only recalibration',
            'evaluation': 'full VAL571 all-session GT, AU round, exact rolling DTW',
            'inference': 'one shared student, ten IDs; no GT, teachers, noise or scaling'})
        stage('generating_train_cache')
        run('cache', [sys.executable, '-m', 'coreset_reactor.teacher_cache',
                     '--data-root', str(args.data_root), '--teacher-checkpoints',
                     *map(str, teachers), '--output-dir', str(args.run_root / 'train_cache'),
                     '--device', f'cuda:{args.gpus[0]}', '--batch-size', '4', '--workers', '4'])
        cache = args.run_root / 'train_cache/manifest.json'
        if read(cache)['teacher_checkpoint_hashes'] != hashes:
            raise ValueError('Teacher changed during cache generation')
        cache_audit = validate_cache_views(cache, args.old_student_run / 'train_cache/manifest.json')
        write(args.run_root / 'cache_view_audit.json', cache_audit)
        stage('training_then_full_val_evaluation', **cache_audit)
        run('students', [sys.executable, '-m', 'coreset_reactor.run_t1_student_distillation',
                        '--teacher-root', str(teacher_root), '--run-root', str(args.run_root),
                        '--data-root', str(args.data_root), '--asset-mam-root', str(args.asset_mam_root),
                        '--target-cache', str(args.target_cache), '--gpus', *map(str, args.gpus),
                        '--arms', 'basic', 'contrast'])
        stage('validating_matched_comparison')
        results, old_results = {}, {}
        for arm in ('basic', 'contrast'):
            new_prov = read(args.run_root / arm / 'provenance.json')
            old_prov = read(args.old_student_run / arm / 'provenance.json')
            for key in ('initial_student_state_sha256', 'observed_source_schedule_sha256',
                        'initialization_checkpoint_hash'):
                if new_prov[key] != old_prov[key]:
                    raise ValueError(f'Old/new matched comparison failed: {arm}/{key}')
            config = read(args.run_root / arm / 'config.json')
            for key in ('seed', 'epochs', 'batch_size', 'lr', 'max_steps',
                        'expert_embedding_dim', 'relation_grad_ratio', 'relation_objective'):
                if config[key] != read(args.old_student_run / arm / 'config.json')[key]:
                    raise ValueError(f'Old/new training mismatch: {arm}/{key}')
            history = read(args.run_root / arm / 'train_metrics.json')
            if history[-1]['step'] != 8300:
                raise ValueError(f'Wrong training budget: {arm}')
            new = read(args.run_root / arm / 'val571_full.json')
            old = read(args.old_student_run / arm / 'val571_full.json')
            validate_result_alignment(new, old)
            validate_result_alignment(new, teacher_eval)
            results[arm], old_results[arm] = new['metrics'], old['metrics']
        summary = {'matched_checks': 'passed', 'teacher': teacher_eval['metrics'],
                   'new_students': results, 'old_students': old_results,
                   'wall_seconds': time.time() - started}
        write(args.run_root / 'teacher_upgrade_comparison.json', summary)
        lines = ['# Mode-reference teacher distillation', '',
                 'REACT2025, full VAL571, all-session GT, official AU rounding and exact FRD.',
                 'Single seed; exploratory VAL, not official sampled-ten TEST.', '',
                 '| Model | FRC | exact FRD | FRDiv | FRVar |',
                 '|---|---:|---:|---:|---:|']
        for name, metrics in [('New teacher', teacher_eval['metrics']),
                              *[(f'Old {a}', m) for a, m in old_results.items()],
                              *[(f'New {a}', m) for a, m in results.items()]]:
            lines.append(f"| {name} | {metrics['FRC']:.6f} | {metrics['exact_FRD']:.6f} | "
                         f"{metrics['FRDiv']:.6f} | {metrics['FRVar']:.6f} |")
        lines += ['', 'Same initialization, all 3320 source/crop views and 8300-update schedule verified.',
                  'Basic KD and signed centered-residual KD; weights/scales calibrated on TRAIN only.',
                  'The student has 3,803,201 parameters; ten outputs without teachers at inference.',
                  'No inference scaling, teacher averaging or GT input. Old weights are retained.',
                  'Training speed, peak memory and coefficients: per-arm config/provenance JSON.',
                  'Inference timing is diagnostic on shared GPUs, not a controlled speed benchmark.', '']
        (args.run_root / 'MODE_REFERENCE_STUDENT.md').write_text('\n'.join(lines))
        stage('complete', summary=summary)
    except Exception as error:
        stage('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
