"""Self-match baseline for the exact training GT, before official processing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.evaluate import official_prediction
from regnn.conditional_data import relative_time_resample_reaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--asset-mam-root', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    data = PairedReactionDataset(args.data_root, 'train', 750, crop_mode='center',
                                normalization_dir=args.asset_mam_root/'external/FaceVerse')
    facial = args.data_root/'train/facial-attributes'
    mapping = {a: json.loads((args.frozen_root/f'manifests/{a}/rank_00.json').read_text())['mapping']
               for a in ('T0', 'T1')}
    raw, aligned_cache, rows = {}, {}, []
    for sample in data:
        length, start = int(sample['source_lengths']), int(sample['crop_start'])
        identifier = sample['clip_id']+'.npy'
        row = {'clip_id': sample['clip_id']}
        for arm in mapping:
            scores = []
            for rank, name in enumerate(mapping[arm][identifier]):
                if name not in raw:
                    raw[name] = torch.from_numpy(np.load(facial/name, allow_pickle=False).astype(np.float32))
                if rank == 0:
                    target = raw[name][start:start+length]
                    assert len(target) == length
                else:
                    key = (name, length)
                    if key not in aligned_cache:
                        aligned_cache[key] = relative_time_resample_reaction(raw[name], length)
                    target = aligned_cache[key]
                pred = official_prediction(target[None]).numpy()
                scores.append(float(candidate_frc(target[None].numpy(), pred)[0]))
            row[arm] = scores
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f'Identity self-match {len(rows)}/{len(data)}', flush=True)
    result = {'rows': rows, 'contexts': len(rows),
              'summary': {a: float(np.mean([r[a] for r in rows], axis=0).sum()) for a in mapping},
              'source_sha256': sha256_file(Path(__file__)),
              'note': 'Exact GT used as prediction, AU rounded; constant channels yield CCC zero, not one.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
