"""Launch the one-weight T1H matched teacher pilot and full VAL evaluation."""
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
    for name in ('root', 'data-root', 'asset-mam-root', 'frozen-root', 'convergence-root',
                 'target-cache', 'processed-target-cache', 'previous-research-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[1, 5, 6])
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError('GPUs must be nonempty and distinct')
    metadata = json.loads(args.processed_target_cache.with_suffix('.json').read_text())
    if metadata['split'] != 'train' or metadata['manifest_sha256'] != sha256_file(args.frozen_root/'manifests/T1/rank_00.json'):
        raise ValueError('Wrong processed target cache / T1 manifest')
    source_files = [
        'coreset_reactor/run_hybrid_teacher_pilot.py', 'coreset_reactor/train_hybrid_teacher.py',
        'coreset_reactor/analyze_hybrid_teacher.py', 'coreset_reactor/results/HYBRID_TEACHER_PROTOCOL.md',
        'coreset_reactor/eval_expert_student_fast.py', 'coreset_reactor/fast_frc.py',
        'coreset_reactor/diagnose_teacher_target_fit.py', 'coreset_reactor/processed_teacher_targets.py',
        'mam_reactor/regnn/train_conditional_regnn.py', 'mam_reactor/regnn/conditional_model.py',
        'mam_reactor/regnn/conditional_data.py']
    reference = args.convergence_root/'arms/T1/rank_00/conditional-step000600-seed1.json'
    expected = json.loads(reference.read_text())
    mapping = json.loads((args.frozen_root/'manifests/T1/rank_00.json').read_text())['mapping']
    for rank in range(1, 10):
        assert json.loads((args.frozen_root/f'manifests/T1/rank_{rank:02d}.json').read_text())['mapping'] == mapping
    args.root.mkdir(parents=True, exist_ok=False)
    provenance = {
        'arm': 'T1H', 'seed': 1, 'steps_per_teacher': 600, 'auxiliary_weight': 0.1,
        'base_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': {name: sha256_file(ROOT/name) for name in source_files},
        'processed_target_cache_sha256': sha256_file(args.processed_target_cache),
        'target_manifest_sha256': {str(r): sha256_file(args.frozen_root/f'manifests/T1/rank_{r:02d}.json') for r in range(10)},
        'q0_checkpoint_sha256': sha256_file(reference.with_suffix('.pth')),
        'new_optimizer_updates': 5400, 'ensemble_budget_including_q0': 6000,
        'loss': 'unchanged raw loss + 0.10 * (processed MSE + 1 - processed CCC)',
        'extra_compute': 'one extra inexpensive loss calculation; no extra model forward, SoftDTW, or target identity',
        'matched_training_checks': 'pending',
        'paths': {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        'strict_guard': 'FRC>=T1, FRD<=T1, FRDiv>=T1, raw-target diagnostic FRC>=T1'}
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    env = dict(os.environ, MAM_REACTOR_ROOT=str(args.asset_mam_root),
               PYTHONPATH=f'{ROOT/"mam_reactor"}:{ROOT}', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def execute(command, gpu, log):
        with log.open('x') as handle:
            subprocess.run(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    def train(gpu, ranks):
        for rank in ranks:
            directory = args.root/f'rank_{rank:02d}'
            directory.mkdir()
            command = [sys.executable, '-m', 'coreset_reactor.train_hybrid_teacher',
                       '--processed-target-cache', str(args.processed_target_cache),
                       '--processed-auxiliary-weight', '0.1', '--data-dir', str(args.data_root),
                       '--run-dir', str(directory), '--epochs', '50', '--batch-size', '32',
                       '--workers', '8', '--precision', 'bf16', '--seed', '1', '--fixed-pair-seed', '1',
                       '--fixed-pair-rank', str(rank), '--fixed-pair-count', '10',
                       '--fixed-pair-manifest', str(args.frozen_root/f'manifests/T1/rank_{rank:02d}.json'),
                       '--fixed-pair-alignment-policy', 'relative_time_masked', '--graph-layers', '2',
                       '--prediction-head', 'linear', '--max-train-steps', '600', '--snapshot-steps', '600',
                       '--save-every', '5', '--print-every', '100']
            (directory/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            execute(command, gpu, directory/'train.log')
            audit = json.loads((directory/'conditional-step000600-seed1.json').read_text())
            for key in ('initial_state_sha256', 'observed_source_schedule_sha256', 'global_step', 'parameter_count'):
                if audit[key] != expected[key]:
                    raise ValueError(f'Matched audit failed for rank {rank}: {key}')
            print(f'T1H rank {rank}: complete; matched audit passed', flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        tasks = [pool.submit(train, gpu, list(range(1, 10))[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        for task in tasks:
            task.result()
    provenance['matched_training_checks'] = 'passed'
    provenance['training_wall_seconds'] = time.monotonic()-started
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    checkpoints = [reference.with_suffix('.pth')] + [
        args.root/f'rank_{r:02d}/conditional-step000600-seed1.pth' for r in range(1, 10)]
    output = args.root/'val571_full.json'
    command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
               '--teacher-checkpoints', *map(str, checkpoints), '--data-root', str(args.data_root),
               '--asset-mam-root', str(args.asset_mam_root), '--device', 'cuda:0', '--eval-seed', '1234',
               '--output', str(output), '--target-cache', str(args.target_cache), '--exact-frd', '--metric-workers', '16']
    execute(command, args.gpus[0], args.root/'val571_full.log')
    print(json.dumps(json.loads(output.read_text())['metrics']), flush=True)
    # Recheck code before analysis; no silently changing the predeclared contract.
    for name, digest in provenance['source_hashes'].items():
        assert sha256_file(ROOT/name) == digest, name
    command = [sys.executable, '-m', 'coreset_reactor.analyze_hybrid_teacher',
               '--root', str(args.root), '--report', str(ROOT/'coreset_reactor/results/HYBRID_TEACHER.md')]
    execute(command, args.gpus[0], args.root/'analysis.log')
    print((args.root/'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
