"""Build frozen TRAIN-only VAE-processed specialist targets for a T1 ablation.

Only the representation changes. GT identities, paired anchor, temporal loss
budget and speaker inputs remain fixed. Seeds are per (GT ID, output length),
not the context-dependent stochastic seed used by evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.evaluate import _deterministic_postprocess


def cache_key(identifier, length):
    return f'{identifier}|{length}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    from framework.modules.post_processor import Processor
    processor = Processor(cfg_dir=str(args.asset_mam_root),
                          ckpt_dir=str(args.asset_mam_root/'pretrained_models/post_processor'),
                          clip_len_test=1000, device=device, num_preds=10)
    doc = json.loads(args.manifest.read_text())
    if doc['split'] != 'train':
        raise ValueError('TRAIN-only target cache required')
    facial = args.data_root/'train/facial-attributes'
    lengths, requests, file_hashes = {}, set(), {}
    for source, targets in doc['mapping'].items():
        sizes = [np.load(args.data_root/'train'/part/source, mmap_mode='r', allow_pickle=False).shape[0]
                 for part in ('facial-attributes', 'audio-features', 'coefficients')]
        length = min(750, *sizes)
        lengths[source] = length
        for target in targets[1:]:
            path = Path(target)
            if path.is_absolute() or '..' in path.parts or len(path.parts) != 3:
                raise ValueError('Invalid TRAIN target path')
            requests.add((target, length))
    cache = {}
    with torch.inference_mode():
        for target, length in sorted(requests):
            raw = torch.from_numpy(np.load(facial/target, allow_pickle=False).astype(np.float32))
            key = cache_key(target, length)
            value = _deterministic_postprocess(processor, torch.zeros(10, length, 25),
                                               [raw], device, 1234, f'TRAIN-teacher-cache/{key}')[0].cpu()
            assert value.shape == (length, 25) and torch.isfinite(value).all()
            cache[key] = value
            file_hashes[target] = sha256_file(facial/target)
            if len(cache) % 50 == 0 or len(cache) == len(requests):
                print(f'Processed fixed targets {len(cache)}/{len(requests)}', flush=True)
    metadata = {'split': 'train', 'contexts': len(lengths), 'targets': len(cache),
                'manifest_sha256': sha256_file(args.manifest),
                'postprocessor_sha256': sha256_file(args.asset_mam_root/'pretrained_models/post_processor/checkpoint.pth'),
                'source_sha256': sha256_file(Path(__file__)), 'GT_file_hashes': file_hashes,
                'seed_contract': 'seed1234, fixed per TRAIN GT identity/output length; no VAL inputs',
                'q0': 'native paired target, no change',
                'ablation': 'same T1 GT IDs; VAE targets replace relative-time targets for q1..q9'}
    torch.save({'metadata': metadata, 'source_lengths': lengths, 'targets': cache}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'GT_file_hashes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
