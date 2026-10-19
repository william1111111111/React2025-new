"""Matched 600-step T2R pilot plus three-shard full TRAIN fit diagnosis."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

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
    run = args.root/'pilot600'
    run.mkdir(parents=True, exist_ok=False)
    expected = json.loads((args.convergence_root/'arms/T0/rank_00/conditional-step000600-seed1.json').read_text())
    files = ['run_supported_teacher_pilot.py', 'supported_teacher_manifest.py',
             'diagnose_teacher_target_fit.py', 'eval_expert_student_fast.py', 'fast_frc.py']
    source_hashes = {f'coreset_reactor/{f}': sha256_file(ROOT/'coreset_reactor'/f) for f in files}
    for name in ('train_conditional_regnn.py', 'conditional_data.py', 'conditional_model.py'):
        source_hashes[f'mam_reactor/regnn/{name}'] = sha256_file(ROOT/'mam_reactor/regnn'/name)
    manifest_dir = args.root/'manifests/T2R'
    manifests = {str(i): sha256_file(manifest_dir/f'rank_{i:02d}.json') for i in range(10)}
    original_mapping = json.loads((args.frozen_root/'manifests/T0/rank_00.json').read_text())['mapping']
    for rank in range(10):
        mapping = json.loads((manifest_dir/f'rank_{rank:02d}.json').read_text())['mapping']
        assert set(mapping) == set(original_mapping)
        assert all(v[0] == original_mapping[k][0] for k, v in mapping.items())
    provenance = {'base_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': source_hashes, 'manifest_hashes': manifests,
                  'arm': 'T2R', 'seed': 1, 'steps': 600, 'batch_size': 32,
                  'q0': 'reuse exact T0_600 frozen paired anchor; target mapping verified identical',
                  'new_optimizer_updates': 9*600, 'ensemble_training_budget_including_reused_q0': 10*600,
                  'formal_schedule_epochs': 50, 'gpus': [1, 5, 6],
                  'evaluation': 'full VAL571 deterministic all-session; unchanged evaluator and GT cache',
                  'predeclared_primary_guard': 'FRC >= matched T0_600 and exact_FRD <= T0_600 and FRDiv > T0_600',
                  'selection_policy': 'frozen support_weight=.5, reuse_weight=.5; no VAL-based sweep',
                  'latency_caveat': 'concurrent diagnostics; throughput is not an isolated benchmark'}
    (run/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    env0 = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
                PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def execute(command, gpu, log):
        with log.open('x') as f:
            subprocess.run(command, cwd=ROOT, env=dict(env0, CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    def diagnose(shard, gpu):
        output = args.root/f'train_fit_shard{shard}.json'
        command = [sys.executable, '-m', 'coreset_reactor.diagnose_teacher_target_fit',
                   '--data-root', str(args.data_root), '--asset-mam-root', str(args.asset_mam_root),
                   '--frozen-root', str(args.frozen_root), '--convergence-root', str(args.convergence_root),
                   '--output', str(output), '--shard-index', str(shard), '--shard-count', '3']
        execute(command, gpu, output.with_suffix('.log'))
        print(f'TRAIN diagnostic shard {shard} completed', flush=True)

    def train_queue(gpu, ranks):
        for rank in ranks:
            directory = run/f'rank_{rank:02d}'
            directory.mkdir()
            command = [sys.executable, '-m', 'regnn.train_conditional_regnn',
                       '--data-dir', str(args.data_root), '--run-dir', str(directory),
                       '--epochs', '50', '--batch-size', '32', '--workers', '8', '--precision', 'bf16',
                       '--seed', '1', '--fixed-pair-seed', '1', '--fixed-pair-rank', str(rank),
                       '--fixed-pair-count', '10', '--fixed-pair-manifest', str(manifest_dir/f'rank_{rank:02d}.json'),
                       '--fixed-pair-alignment-policy', 'relative_time_masked',
                       '--graph-layers', '2', '--prediction-head', 'linear',
                       '--max-train-steps', '600', '--snapshot-steps', '600', '--save-every', '5', '--print-every', '100']
            (directory/'command.json').write_text(json.dumps(command, indent=2))
            execute(command, gpu, directory/'train.log')
            audit = json.loads((directory/'conditional-step000600-seed1.json').read_text())
            for key in ('initial_state_sha256', 'observed_source_schedule_sha256', 'global_step', 'parameter_count'):
                if audit[key] != expected[key]:
                    raise ValueError(f'Matched audit failed: rank{rank}/{key}')
            print(f'T2R rank {rank}: trained, matched audit passed', flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        diagnostic_tasks = [pool.submit(diagnose, shard, gpu) for shard, gpu in enumerate((1, 5, 6))]
        training_tasks = [pool.submit(train_queue, 5, list(range(1, 10, 2))),
                          pool.submit(train_queue, 6, list(range(2, 10, 2)))]
        for task in training_tasks:
            task.result()
        provenance['matched_training_checks'] = 'passed'
        provenance['training_wall_seconds_with_concurrent_diagnostics'] = time.monotonic()-started
        (run/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        checkpoints = [args.convergence_root/'arms/T0/rank_00/conditional-step000600-seed1.pth'] + [
            run/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1, 10)]
        for suffix in ('fast', 'full'):
            output = run/f'val571_{suffix}.json'
            command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
                       '--teacher-checkpoints', *map(str, checkpoints), '--data-root', str(args.data_root),
                       '--asset-mam-root', str(args.asset_mam_root), '--device', 'cuda:0',
                       '--eval-seed', '1234', '--output', str(output), '--target-cache', str(args.target_cache)]
            if suffix == 'full':
                command += ['--exact-frd', '--metric-workers', '16']
            execute(command, 5, output.with_suffix('.log'))
            print(json.dumps({'T2R_eval': suffix, 'metrics': json.loads(output.read_text())['metrics']}), flush=True)
        for task in diagnostic_tasks:
            task.result()

    shards = [json.loads((args.root/f'train_fit_shard{i}.json').read_text()) for i in range(3)]
    for shard in shards[1:]:
        assert shard['hashes'] == shards[0]['hashes']
        assert shard['source_sha256'] == shards[0]['source_sha256']
    rows = sorted([r for shard in shards for r in shard['rows']], key=lambda r: r['clip_id'])
    expected_ids = {(p.relative_to(args.data_root/'train/facial-attributes').with_suffix('')).as_posix()
                    for p in (args.data_root/'train/facial-attributes/speaker').glob('*/*.npy')}
    assert len(rows) == len(expected_ids) == len({r['clip_id'] for r in rows})
    assert {r['clip_id'] for r in rows} == expected_ids
    fit_summary = {}
    for label in rows[0]['arms']:
        fit_summary[label] = {}
        for key in rows[0]['arms'][label]:
            mean = np.mean([r['arms'][label][key] for r in rows], axis=0)
            fit_summary[label][key] = float(mean.sum()) if np.ndim(mean) else float(mean)
            if np.ndim(mean):
                fit_summary[label][key+'_per_rank'] = mean.tolist()
    full_fit = {**{k: v for k, v in shards[0].items() if k not in ('rows', 'summary', 'contexts', 'shard_index', 'seconds')},
                'rows': rows, 'summary': fit_summary, 'contexts': len(rows), 'shards_merged': 3}
    (args.root/'train_fit_full.json').write_text(json.dumps(full_fit, indent=2, allow_nan=False)+'\n')
    result = json.loads((run/'val571_full.json').read_text())
    reference = json.loads((args.convergence_root/'eval/T0_s000600_val571_full.json').read_text())
    for key in ('clip_id', 'target_cache_key'):
        assert [r[key] for r in result['rows']] == [r[key] for r in reference['rows']]
    a, b = result['metrics'], reference['metrics']
    summary = {'metrics': a, 'delta_vs_T0_600': {k: a[k]-b[k] for k in ('FRC', 'exact_FRD', 'FRDiv', 'FRVar')},
               'strict_guard_pass': a['FRC'] >= b['FRC'] and a['exact_FRD'] <= b['exact_FRD'] and a['FRDiv'] > b['FRDiv'],
               'matched_training_checks': 'passed', 'matched_evaluation_checks': 'passed',
               'TRAIN_diagnostic_contexts': len(rows), 'wall_seconds': time.monotonic()-started}
    (run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
