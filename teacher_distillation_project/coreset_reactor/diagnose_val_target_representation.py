"""Post-hoc raw-target FRC alongside the unchanged official-processed metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.fast_frc import candidate_frc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asset-mam-root', type=Path, required=True)
    args = p.parse_args()
    output = args.root/'val_representation_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    files = {'T0': args.convergence_root/'eval/T0_s000600_val571_full.json',
             'T1': args.convergence_root/'eval/T1_s000600_val571_full.json',
             'T2R': args.root/'pilot600/val571_full.json',
             'T1P': args.root/'processed_pilot600/val571_full.json'}
    records = {a: json.loads(path.read_text()) for a, path in files.items()}
    prediction = {a: torch.load(path.with_suffix('.predictions.pt'), map_location='cpu') for a, path in files.items()}
    data = PairedReactionDataset(args.data_root, 'val', 750, crop_mode='center',
                                normalization_dir=args.asset_mam_root/'external/FaceVerse')
    facial = args.data_root/'val/facial-attributes'
    pools = {}
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools.setdefault(path.parent.name, []).append(path)
    rows = []
    for index in range(len(data)):
        sample = data[index]
        length, start = int(sample['source_lengths']), int(sample['crop_start'])
        paired = (facial/sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
        paths = [paired, *[x for x in pools[sample['session_id']] if x != paired]]
        raw = [torch.from_numpy(np.load(x, allow_pickle=False).astype(np.float32)) for x in paths]
        target = aligned_targets(raw, length, start).numpy()
        row = {'clip_id': sample['clip_id'], 'arms': {}}
        for arm in files:
            assert prediction[arm]['clip_ids'][index] == records[arm]['rows'][index]['clip_id'] == sample['clip_id']
            pred = prediction[arm]['official_predictions'][index].numpy()
            row['arms'][arm] = {
                'all_session_training_representation_FRC': float(candidate_frc(target, pred).sum()),
                'native_paired_raw_FRC': float(candidate_frc(target[:1], pred).sum()),
                'all_session_official_processed_FRC': float(sum(records[arm]['rows'][index]['frc']))}
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f'VAL target representation {len(rows)}/{len(data)}', flush=True)
    summary = {arm: {key: float(np.mean([r['arms'][arm][key] for r in rows]))
                     for key in rows[0]['arms'][arm]} for arm in files}
    output.write_text(json.dumps({'rows': rows, 'summary': summary, 'contexts': len(rows),
                                'source_sha256': sha256_file(Path(__file__)),
                                'purpose': 'post-hoc interpretation only; raw-target scores are not official metrics'}, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
