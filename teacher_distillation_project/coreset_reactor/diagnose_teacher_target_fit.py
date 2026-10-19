"""Full TRAIN speaker-side fixed-target and all-session FRC diagnostics.

Keeps training-aligned and official-processed targets separate. No training,
VAL-target injection, or claim that TRAIN and VAL contain identical reactions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.teacher_cache import load_teacher, sha256_file
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.evaluate import official_prediction, _deterministic_postprocess
from coreset_reactor.fast_frc import candidate_frc
from regnn.conditional_data import relative_time_resample_reaction


def aligned_targets(raw, length, start):
    """The paired item uses source time; cross-listener items use relative time."""
    paired = raw[0][start:start + length]
    if len(paired) != length:
        raise ValueError('Paired valid prefix differs; do not score padded GT')
    return torch.stack([paired, *[relative_time_resample_reaction(x, length) for x in raw[1:]]])


def fixed_scores(target, prediction, indices):
    return np.array([candidate_frc(target[j:j+1], prediction[i:i+1])[0]
                     for i, j in enumerate(indices)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', nargs='+', type=int, default=[600, 1200, 2400])
    p.add_argument('--max-examples', type=int, default=0)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from framework.modules.post_processor import Processor
    processor = Processor(cfg_dir=str(args.asset_mam_root),
                          ckpt_dir=str(args.asset_mam_root/'pretrained_models/post_processor'),
                          clip_len_test=1000, device=device, num_preds=10)
    data = PairedReactionDataset(args.data_root, 'train', 750, crop_mode='center',
                                normalization_dir=args.asset_mam_root/'external/FaceVerse')
    facial = args.data_root/'train/facial-attributes'
    pools = {}
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools.setdefault(path.parent.name, []).append(path)
    manifests, hashes, models = {}, {}, {}
    for arm in ('T0', 'T1'):
        path = args.frozen_root/'manifests'/arm/'rank_00.json'
        manifests[arm] = json.loads(path.read_text())['mapping']
        hashes[f'{arm}_manifest'] = sha256_file(path)
        for step in args.steps:
            label = f'{arm}_{step}'
            paths = [args.convergence_root/'arms'/arm/f'rank_{r:02d}'/
                     f'conditional-step{step:06d}-seed1.pth' for r in range(10)]
            hashes[label] = [sha256_file(x) for x in paths]
            models[label] = [load_teacher(x, device)[0] for x in paths]
    count = len(data) if args.max_examples == 0 else min(args.max_examples, len(data))
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard')
    sample_indices = list(range(count))[args.shard_index::args.shard_count]
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for index in sample_indices:
            sample = data[index]
            length, start = int(sample['source_lengths']), int(sample['crop_start'])
            paired = (facial/sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
            paths = [paired, *[x for x in pools[sample['session_id']] if x != paired]]
            identifiers = [x.relative_to(facial).as_posix() for x in paths]
            lookup = {x: i for i, x in enumerate(identifiers)}
            raw = [torch.from_numpy(np.load(x, allow_pickle=False).astype(np.float32)) for x in paths]
            aligned = aligned_targets(raw, length, start).numpy()
            inputs = [sample[k][None, :length].to(device) for k in
                      ('speaker_audio', 'speaker_emotion', 'speaker_3dmm')]
            inputs.append(torch.tensor([length], device=device))
            processed = None
            row = {'clip_id': sample['clip_id'], 'gt_count': len(paths), 'length': length,
                   'target_length_mismatch_count': sum(len(x) != length for x in raw),
                   'arms': {}}
            for label, ensemble in models.items():
                arm = label.split('_')[0]
                indices = [lookup[x] for x in manifests[arm][sample['clip_id']+'.npy']]
                pred = torch.stack([m(*inputs)['prediction'][0].cpu() for m in ensemble])
                official = official_prediction(pred).numpy()
                if processed is None:
                    processed = _deterministic_postprocess(
                        processor, torch.from_numpy(official), raw, device, 1234,
                        sample['clip_id']).cpu().numpy()
                fixed_raw = fixed_scores(aligned, pred.numpy(), indices)
                fixed_round = fixed_scores(aligned, official, indices)
                raw_all = candidate_frc(aligned, official)
                processed_all = candidate_frc(processed, official)
                fixed_processed = fixed_scores(processed, official, indices)
                # Treat exact training targets as predictions to measure how
                # processing alone changes the matching of the same GT identity.
                identity_pred = official_prediction(torch.from_numpy(aligned[indices])).numpy()
                identity_same = fixed_scores(processed, identity_pred, indices)
                identity_best = candidate_frc(processed, identity_pred)
                assert np.all(raw_all >= fixed_round-1e-8)
                assert np.all(processed_all >= fixed_processed-1e-8)
                row['arms'][label] = {
                    'fixed_training_aligned_soft': fixed_raw.tolist(),
                    'fixed_training_aligned_rounded': fixed_round.tolist(),
                    'all_session_training_aligned_rounded': raw_all.tolist(),
                    'fixed_official_processed': fixed_processed.tolist(),
                    'all_session_official_processed': processed_all.tolist(),
                    'exact_target_to_same_processed_identity': identity_same.tolist(),
                    'exact_target_to_all_processed_GT': identity_best.tolist(),
                    'fixed_target_trajectory_MSE': float(np.mean((pred.numpy()-aligned[indices])**2)),
                }
            rows.append(row)
            if len(rows) % 25 == 0 or len(rows) == len(sample_indices):
                args.output.with_suffix('.progress.json').write_text(json.dumps({
                    'completed': len(rows), 'total': len(sample_indices), 'elapsed_seconds': time.monotonic()-started}))
                print(f'TRAIN target-fit {len(rows)}/{len(sample_indices)}', flush=True)
    summary = {}
    for label in models:
        values = [x['arms'][label] for x in rows]
        summary[label] = {}
        for key in values[0]:
            mean = np.mean([x[key] for x in values], axis=0)
            summary[label][key] = float(mean.sum()) if np.ndim(mean) else float(mean)
            if np.ndim(mean):
                summary[label][key+'_per_rank'] = mean.tolist()
    result = {'summary': summary, 'rows': rows, 'hashes': hashes,
              'source_sha256': sha256_file(Path(__file__)), 'contexts': len(rows),
              'shard_index': args.shard_index, 'shard_count': args.shard_count,
              'protocol': 'TRAIN speaker-side; deterministic center750; three explicit target/rounding contracts',
              'limitations': ['training-mode dropout disabled', 'TRAIN is not held-out performance',
                              'reverse-role TRAIN examples not included; matches VAL speaker-input direction',
                              'same-identity score is not a perfect-score ceiling because constant channels yield CCC=0'],
              'postprocessor_sha256': sha256_file(args.asset_mam_root/'pretrained_models/post_processor/checkpoint.pth'),
              'seconds': time.monotonic()-started}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
