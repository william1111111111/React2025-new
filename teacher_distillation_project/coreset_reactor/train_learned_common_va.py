"""Fit the frozen TRAIN-only inference-safe common-VA corrector."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from coreset_reactor.learned_common_va import CommonVACorrector
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import sha256_file


SEED = 20260925


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_labels(paths):
    shards, summaries = [], []
    for path in paths:
        shards.append(torch.load(path/'labels.pt', map_location='cpu'))
        summary = json.loads((path/'summary.json').read_text())
        if sha256_file(path/'labels.pt') != summary['labels_sha256']:
            raise ValueError(f'Label hash mismatch: {path}')
        summaries.append(summary)
    order = sorted(range(len(shards)), key=lambda i: shards[i]['context_start'])
    shards, summaries = [shards[i] for i in order], [summaries[i] for i in order]
    cursor = 0
    for shard in shards:
        if shard['split'] != 'train' or shard['crop_mode'] != 'center':
            raise ValueError('Labels must be TRAIN center crops')
        if shard['context_start'] != cursor or shard['context_stop']-cursor != len(shard['clip_ids']):
            raise ValueError('Label shards are not complete and contiguous')
        cursor = shard['context_stop']
    if cursor != 1660:
        raise ValueError(f'Expected 1660 labels, got {cursor}')
    for key in ('checkpoint_sha256', 'postprocessor_checkpoint_sha256', 'source_hashes'):
        if any(summary[key] != summaries[0][key] for summary in summaries[1:]):
            raise ValueError(f'Inconsistent label provenance: {key}')
    for key in ('teacher_mean', 'teacher_std', 'target_shift', 'lengths', 'gate'):
        shards[0][key] = torch.cat([x[key] for x in shards])
    shards[0]['clip_ids'] = sum([x['clip_ids'] for x in shards], [])
    return shards[0], summaries


class CorrectorDataset(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
        if len(data) != len(labels['clip_ids']):
            raise ValueError('Dataset/label length mismatch')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if item['clip_id'] != self.labels['clip_ids'][index]:
            raise ValueError(f'Label identity mismatch at {index}')
        if int(item['source_lengths']) != int(self.labels['lengths'][index]):
            raise ValueError(f'Label length mismatch at {index}')
        return {key: item[key] for key in ('speaker_audio', 'speaker_emotion', 'speaker_3dmm',
                                           'source_lengths', 'clip_id')} | {
            'teacher_mean': self.labels['teacher_mean'][index].float(),
            'teacher_std': self.labels['teacher_std'][index].float(),
            'target_shift': self.labels['target_shift'][index].float(),
            'gate': self.labels['gate'][index].float()}


def masked_smooth_l1(prediction, target, mask, beta=.02):
    value = F.smooth_l1_loss(prediction, target, reduction='none', beta=beta)
    weight = mask[..., None].to(value)
    return (value*weight).sum()/(weight.sum()*value.shape[-1]).clamp_min(1)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, nargs='+', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.epochs != 20 or args.batch_size != 8:
        raise ValueError('Frozen protocol requires 20 epochs and batch size 8')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Frozen protocol requires CUDA BF16 training')
    torch.cuda.set_device(device)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    labels, summaries = load_labels(args.labels)
    selection_counts = Counter()
    for summary in summaries:
        selection_counts.update(summary['selection_counts'])
    label_oracle = {
        'selection_counts': dict(selection_counts),
        'gate_positive_fraction': float(labels['gate'].float().mean()),
        'mean_FRC_delta': float(sum(x['contexts']*x['mean_FRC_delta'] for x in summaries)/len(labels['gate'])),
        'mean_exact_FRD_delta': float(sum(x['contexts']*x['mean_FRD_delta'] for x in summaries)/len(labels['gate']))}
    base = PairedReactionDataset(args.data_root, 'train', 750, crop_mode='center',
                                 normalization_dir=args.asset_mam_root/'external/FaceVerse')
    dataset = CorrectorDataset(base, labels)
    model = CommonVACorrector().to(device)
    initial_sha = state_hash(model.state_dict())
    parameter_count = sum(x.numel() for x in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(SEED)
    order_digest = hashlib.sha256()
    history, started = [], time.monotonic()

    for epoch in range(args.epochs):
        order = torch.randperm(len(dataset), generator=generator).tolist()
        order_digest.update(np.asarray(order, dtype=np.int32).tobytes())
        loader = DataLoader(Subset(dataset, order), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=False,
                            worker_init_fn=seed_worker)
        model.train()
        sums = {'loss': 0., 'shift': 0., 'velocity': 0., 'gate': 0., 'batches': 0}
        for batch in loader:
            keys = ('speaker_audio', 'speaker_emotion', 'speaker_3dmm', 'teacher_mean',
                    'teacher_std', 'target_shift', 'source_lengths', 'gate')
            audio, emotion, face, mean, std, target, lengths, gate = [batch[k].to(device, non_blocking=True) for k in keys]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(audio, emotion, face, mean, std, lengths)
                shift_loss = masked_smooth_l1(out['shift'], target, out['valid_mask'])
                velocity_mask = out['valid_mask'][:, 1:] & out['valid_mask'][:, :-1]
                velocity_loss = masked_smooth_l1(out['shift'][:, 1:]-out['shift'][:, :-1],
                                                 target[:, 1:]-target[:, :-1], velocity_mask)
                gate_loss = F.binary_cross_entropy_with_logits(out['gate_logit'], gate)
                loss = shift_loss+.1*velocity_loss+.2*gate_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            for name, value in [('loss', loss), ('shift', shift_loss),
                                ('velocity', velocity_loss), ('gate', gate_loss)]:
                sums[name] += float(value.detach())
            sums['batches'] += 1
        scheduler.step()
        row = {'epoch': epoch+1, 'lr': optimizer.param_groups[0]['lr'],
               **{k: v/sums['batches'] for k, v in sums.items() if k != 'batches'},
               'elapsed_seconds': time.monotonic()-started}
        history.append(row)
        with (args.output/'train.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    model.eval()
    checkpoint = args.output/'common-va-corrector-epoch0020-seed20260925.pth'
    train_seconds = time.monotonic()-started
    payload = {'model': model.state_dict(), 'model_config': {'hidden': 64, 'dropout': .1, 'max_shift': .35},
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'epoch': args.epochs, 'seed': SEED, 'parameter_count': parameter_count,
               'train_seconds': train_seconds,
               'train_label_oracle_summary': label_oracle,
               'initial_state_sha256': initial_sha, 'source_order_sha256': order_digest.hexdigest(),
               'label_shards': [str(x.resolve()) for x in args.labels],
               'label_sha256': [sha256_file(x/'labels.pt') for x in args.labels]}
    torch.save(payload, checkpoint)
    summary = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
               'parameter_count': parameter_count, 'initial_state_sha256': initial_sha,
               'source_order_sha256': order_digest.hexdigest(), 'seed': SEED,
               'epochs': args.epochs, 'batch_size': args.batch_size,
               'optimizer': 'AdamW(lr=3e-4,betas=(0.9,0.99),weight_decay=1e-4)',
               'scheduler': 'CosineAnnealingLR(T_max=20), stepped after each epoch',
               'precision': 'BF16 autocast; FP32 optimizer', 'gradient_clip': 1.0,
               'train_seconds': train_seconds,
               'training_context_exposures': args.epochs*len(dataset),
               'training_context_exposures_per_second': args.epochs*len(dataset)/train_seconds,
               'gate_positive_fraction': label_oracle['gate_positive_fraction'],
               'train_label_oracle_summary': label_oracle,
               'history': history, 'label_summaries': summaries,
               'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('history','label_summaries')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
