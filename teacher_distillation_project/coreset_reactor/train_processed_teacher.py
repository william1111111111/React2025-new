"""Isolated target-representation adapter around the unchanged Mam trainer."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.processed_teacher_targets import cache_key
from regnn.conditional_data import ConditionalReactionDataset
from regnn import train_conditional_regnn as trainer

TARGET_CACHE = None


class ProcessedTargetDataset(ConditionalReactionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fixed_pair_rank is None or self.fixed_pair_alignment_policy != 'relative_time_masked':
            raise ValueError('Requires existing fixed-pair relative-time baseline')
        if TARGET_CACHE is None or TARGET_CACHE['metadata']['split'] != 'train':
            raise ValueError('TRAIN processed-target cache is missing')
        self.processed_cache = TARGET_CACHE

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.fixed_pair_rank == 0:
            return result
        identifier = result['clip_id']+'.npy'
        length = int(result['length'])
        if self.processed_cache['source_lengths'][identifier] != length:
            raise ValueError('Unexpected source view length; rebuild processed target cache')
        target_id = self.fixed_pair_mapping[identifier][self.fixed_pair_rank]
        target = self.processed_cache['targets'][cache_key(target_id, length)]
        result['target'] = torch.zeros_like(result['target'])
        result['target'][:length] = target
        return result


def main():
    global TARGET_CACHE
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--processed-target-cache', type=Path, required=True)
    args, remaining = p.parse_known_args()
    TARGET_CACHE = torch.load(args.processed_target_cache, map_location='cpu')
    original_parse = trainer.parse_args

    def parse_with_provenance():
        value = original_parse()
        value.processed_target_cache = str(args.processed_target_cache.resolve())
        value.processed_target_cache_sha256 = sha256_file(args.processed_target_cache)
        value.specialist_target_representation = 'frozen TRAIN official VAE processing'
        return value

    trainer.parse_args = parse_with_provenance
    trainer.ConditionalReactionDataset = ProcessedTargetDataset
    sys.argv = [sys.argv[0], *remaining]
    trainer.main()


if __name__ == '__main__':
    main()
