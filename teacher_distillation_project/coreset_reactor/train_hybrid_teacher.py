"""Keep native fixed-GT supervision and add a weak processed-target loss.

The private 50-channel target transport is split inside the loss; model outputs
remain 25-D. No historical trainer or dataset implementation is modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from coreset_reactor.processed_teacher_targets import cache_key
from coreset_reactor.teacher_cache import sha256_file
from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import ConditionalREGNNLoss
from regnn import train_conditional_regnn as trainer

TARGET_CACHE = None


class HybridTargetDataset(ConditionalReactionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fixed_pair_rank not in range(1, 10):
            raise ValueError('Specialists only; reuse the unchanged native paired anchor')
        if self.fixed_pair_alignment_policy != 'relative_time_masked':
            raise ValueError('Requires the frozen relative-time baseline')
        if TARGET_CACHE is None or TARGET_CACHE['metadata']['split'] != 'train':
            raise ValueError('Requires a TRAIN-only processed target cache')
        self.processed_cache = TARGET_CACHE

    def __getitem__(self, index):
        result = super().__getitem__(index)
        identifier, length = result['clip_id']+'.npy', int(result['length'])
        if self.processed_cache['source_lengths'][identifier] != length:
            raise ValueError('Source length differs from processed-target cache')
        target_id = self.fixed_pair_mapping[identifier][self.fixed_pair_rank]
        processed = self.processed_cache['targets'][cache_key(target_id, length)]
        auxiliary = torch.zeros_like(result['target'])
        auxiliary[:length] = processed
        result['target'] = torch.cat((result['target'], auxiliary), dim=-1)
        return result


class HybridTargetLoss(ConditionalREGNNLoss):
    def __init__(self, ccc_weight=1.0, velocity_weight=0.05, auxiliary_weight=0.1,
                 audit_every=0):
        super().__init__(ccc_weight, velocity_weight)
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('Auxiliary weight must be finite and nonnegative')
        self.auxiliary_weight = auxiliary_weight
        self.auxiliary = ConditionalREGNNLoss(ccc_weight=ccc_weight, velocity_weight=0.0)
        self.audit_every, self.calls = audit_every, 0

    def forward(self, prediction, target, valid_mask, velocity_supervision=None):
        if prediction.shape[-1] != 25 or target.shape[-1] != 50:
            raise ValueError('Expected 25-D prediction and packed raw/processed target')
        raw, processed = target.split(25, dim=-1)
        losses = super().forward(prediction, raw, valid_mask, velocity_supervision)
        raw_loss = losses['loss']
        auxiliary_loss = prediction.new_zeros(())
        if self.auxiliary_weight:
            no_velocity = torch.zeros(prediction.shape[0], dtype=torch.bool, device=prediction.device)
            auxiliary_loss = self.auxiliary(prediction, processed, valid_mask, no_velocity)['loss']
            losses['loss'] = raw_loss + self.auxiliary_weight * auxiliary_loss
        self.calls += 1
        if self.audit_every and self.calls % self.audit_every == 0:
            print(json.dumps({'hybrid_loss_step': self.calls,
                              'raw_loss': float(raw_loss.detach()),
                              'processed_auxiliary_loss': float(auxiliary_loss.detach()),
                              'auxiliary_weight': self.auxiliary_weight}), flush=True)
        return losses


def main():
    global TARGET_CACHE
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--processed-target-cache', type=Path, required=True)
    p.add_argument('--processed-auxiliary-weight', type=float, default=0.1)
    args, remaining = p.parse_known_args()
    TARGET_CACHE = torch.load(args.processed_target_cache, map_location='cpu')
    original_parse = trainer.parse_args

    def parse_with_provenance():
        value = original_parse()
        if value.listener_3dmm_weight or value.supervision_tail_frames or value.online_rollout_blocks != 1:
            raise ValueError('Pilot only supports the frozen full-context 25-D configuration')
        manifest_hash = sha256_file(value.fixed_pair_manifest)
        # Rank metadata differs, but every rank must retain the exact mapping.
        manifest = json.loads(value.fixed_pair_manifest.read_text())
        if manifest['split'] != 'train':
            raise ValueError('Only TRAIN targets are permitted')
        value.processed_target_cache = str(args.processed_target_cache.resolve())
        value.processed_target_cache_sha256 = sha256_file(args.processed_target_cache)
        value.hybrid_target_manifest_sha256 = manifest_hash
        value.processed_auxiliary_weight = args.processed_auxiliary_weight
        value.specialist_target_representation = 'original raw target + independent weak processed loss'
        return value

    def make_loss(**kwargs):
        return HybridTargetLoss(**kwargs, auxiliary_weight=args.processed_auxiliary_weight, audit_every=100)

    trainer.parse_args = parse_with_provenance
    trainer.ConditionalReactionDataset = HybridTargetDataset
    trainer.ConditionalREGNNLoss = make_loss
    sys.argv = [sys.argv[0], *remaining]
    trainer.main()


if __name__ == '__main__':
    main()
