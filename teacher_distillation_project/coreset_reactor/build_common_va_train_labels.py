"""Build TRAIN-only common-VA oracle labels for the inference-safe corrector."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.common_va import (VARIANTS, feasible_shift, apply_shift,
    optimize_frc_shift, SharedVAQualityPool, strict_choice)
from coreset_reactor.evaluate import _deterministic_postprocess, official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import load_teacher, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_paths(root, arm, step):
    return [root/'arms'/arm/f'rank_{rank:02d}'/f'conditional-step{step:06d}-seed1.pth'
            for rank in range(10)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context-start', type=int, required=True)
    p.add_argument('--context-stop', type=int, required=True)
    p.add_argument('--metric-workers', type=int, default=10)
    p.add_argument('--eval-seed', type=int, default=1234)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('CUDA is required for teacher inference and oracle optimization')
    torch.cuda.set_device(device)
    data = PairedReactionDataset(args.data_root, 'train', 750, crop_mode='center',
                                normalization_dir=args.asset_mam_root/'external/FaceVerse')
    start, stop = args.context_start, args.context_stop
    if not 0 <= start < stop <= len(data):
        raise ValueError(f'Invalid shard [{start},{stop})/{len(data)}')
    t1_paths = checkpoint_paths(args.convergence_root, 'T1', 600)
    t0_paths = checkpoint_paths(args.convergence_root, 'T0', 2400)
    if not all(x.exists() for x in t1_paths+t0_paths):
        raise FileNotFoundError('Frozen teacher checkpoints missing')
    t1 = [load_teacher(path, device)[0].eval() for path in t1_paths]
    t0 = [load_teacher(path, device)[0].eval() for path in t0_paths]
    from framework.modules.post_processor import Processor
    processor = Processor(cfg_dir=str(args.asset_mam_root),
                          ckpt_dir=str(args.asset_mam_root/'pretrained_models/post_processor'),
                          device=device, clip_len_test=1000, num_preds=10)
    facial = args.data_root/'train/facial-attributes'
    pools = defaultdict(list)
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools[path.parent.name].append(path)
    quality_pool = SharedVAQualityPool(args.metric_workers, max(map(len, pools.values())))
    count = stop-start
    teacher_mean = torch.zeros(count, 750, 25, dtype=torch.float16)
    teacher_std = torch.zeros_like(teacher_mean)
    target_shift = torch.zeros(count, 750, 2, dtype=torch.float16)
    lengths = torch.zeros(count, dtype=torch.int16)
    gate = torch.zeros(count, dtype=torch.bool)
    clip_ids, rows = [], []
    digest = hashlib.sha256()
    started = time.monotonic()
    try:
        for local, index in enumerate(range(start, stop)):
            sample = data[index]
            length = int(sample['source_lengths'])
            inputs = [sample[key][None, :length].to(device) for key in
                      ('speaker_audio', 'speaker_emotion', 'speaker_3dmm')]
            size = torch.tensor([length], device=device)
            with torch.inference_mode():
                baseline_soft = torch.stack([model(*inputs, size)['prediction'][0].cpu() for model in t1])
                quality_soft = torch.stack([model(*inputs, size)['prediction'][0].cpu() for model in t0])
                baseline, quality_reference = official_prediction(baseline_soft), official_prediction(quality_soft)
                paired = (facial/sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
                gt_paths = [paired, *[path for path in pools[sample['session_id']] if path != paired]]
                if not paired.exists() or len(gt_paths) != len(pools[sample['session_id']]):
                    raise ValueError(f'Incomplete GT pool: {sample["clip_id"]}')
                raw = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32)) for path in gt_paths]
                target = _deterministic_postprocess(processor, baseline, raw, device,
                                                    args.eval_seed, sample['clip_id']).cpu()
                centroid = feasible_shift(baseline, quality_reference[..., 15:17].mean(0)-baseline[..., 15:17].mean(0))
            # The local 32-parameter oracle is the only gradient-bearing part.
            optimized, objective = optimize_frc_shift(baseline, target, device)
            shifts = torch.stack((torch.zeros_like(centroid), centroid,
                                  .25*optimized, .5*optimized, optimized))
            variants = [apply_shift(baseline, shift) for shift in shifts]
            frc = candidate_frc(target.numpy(), torch.cat(variants).numpy()).reshape(len(VARIANTS), 10)
            frd, calls = quality_pool.score_variants(variants, target)
            choice = strict_choice(frc.sum(1), frd.sum(1))
            assert frc[choice].sum() >= frc[0].sum() and frd[choice].sum() <= frd[0].sum()
            geometry_error = max(float(((value.double()-value.double().mean(0))-
                                        (baseline.double()-baseline.double().mean(0))).abs().max())
                                 for value in variants)
            if geometry_error >= 3e-7:
                raise ValueError('Common geometry changed')
            teacher_mean[local, :length] = baseline.mean(0).half()
            teacher_std[local, :length] = baseline.std(0, unbiased=False).half()
            target_shift[local, :length] = shifts[choice].half()
            lengths[local], gate[local] = length, choice != 0
            clip_ids.append(sample['clip_id'])
            digest.update(json.dumps({'index': index, 'clip_id': sample['clip_id'],
                                      'length': length}, sort_keys=True).encode())
            rows.append({'index': index, 'clip_id': sample['clip_id'], 'session_id': sample['session_id'],
                         'length': length, 'gt_count': len(target), 'choice': VARIANTS[choice],
                         'baseline_FRC': float(frc[0].sum()), 'selected_FRC': float(frc[choice].sum()),
                         'baseline_exact_FRD': float(frd[0].sum()),
                         'selected_exact_FRD': float(frd[choice].sum()),
                         'oracle_objective': objective, 'geometry_max_error': geometry_error,
                         'dtw_group_calls': calls.tolist()})
            if (local+1) % 10 == 0 or local+1 == count:
                progress = {'completed': local+1, 'total': count,
                            'elapsed_seconds': time.monotonic()-started,
                            'last_clip_id': sample['clip_id']}
                (args.output/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    finally:
        quality_pool.close()
    payload = {'teacher_mean': teacher_mean, 'teacher_std': teacher_std,
               'target_shift': target_shift, 'lengths': lengths, 'gate': gate,
               'clip_ids': clip_ids, 'context_start': start, 'context_stop': stop,
               'split': 'train', 'crop_mode': 'center'}
    torch.save(payload, args.output/'labels.pt')
    source_files = ['coreset_reactor/build_common_va_train_labels.py', 'coreset_reactor/common_va.py',
                    'coreset_reactor/results/LEARNED_COMMON_VA_PROTOCOL.md',
                    'coreset_reactor/fast_frc.py', 'coreset_reactor/oracle_select.py',
                    'mam_reactor/tools/compute_frd_resumable.py']
    summary = {'split': 'train', 'contexts': count, 'dataset_total': len(data),
               'context_start': start, 'context_stop': stop,
               'selection_counts': dict(__import__('collections').Counter(r['choice'] for r in rows)),
               'gate_positive_fraction': float(gate.float().mean()),
               'mean_FRC_delta': float(np.mean([r['selected_FRC']-r['baseline_FRC'] for r in rows])),
               'mean_FRD_delta': float(np.mean([r['selected_exact_FRD']-r['baseline_exact_FRD'] for r in rows])),
               'source_schedule_sha256': digest.hexdigest(), 'rows': rows,
               'labels_sha256': sha256_file(args.output/'labels.pt'),
               'checkpoint_sha256': {'T1_600': [sha256_file(x) for x in t1_paths],
                                     'T0_2400': [sha256_file(x) for x in t0_paths]},
               'postprocessor_checkpoint_sha256': sha256_file(args.asset_mam_root/'pretrained_models/post_processor/checkpoint.pth'),
               'source_hashes': {name: sha256_file(ROOT/name) for name in source_files},
               'base_git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'seconds': time.monotonic()-started,
               'protocol': 'TRAIN only; all-session processed GT; finite common-VA oracle; exact rolling FRD'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','checkpoint_sha256','source_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
