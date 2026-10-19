"""TRAIN-only fixed teachers with supported prototypes and reuse penalties.

T2R is distinct from the historical unconstrained-GT oracle named T2.
The nine global prototype identities are reused from the frozen T1 manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from coreset_reactor.dataset import full_reaction_descriptor
from coreset_reactor.dataset_gt_oracle import official_distance_matrix, frdiv
from coreset_reactor.maxfrdiv_teacher_manifest import _load_raw, _relative_resample
from coreset_reactor.teacher_cache import sha256_file


def supported_assignment(values, centers, paired_indices, source_ids,
                         support_weight=.5, reuse_weight=.5):
    """Deterministic rectangular assignment: nine distinct supported targets.

Per-slot reuse is normalized by expected uniform use (contexts / targets).
It is a soft penalty, not a hard cap and not an estimate of compatibility.
"""
    n, modes = len(values), len(centers)
    if n <= modes or len(paired_indices) != len(source_ids):
        raise ValueError('Insufficient distinct targets or missing source IDs')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('Duplicate source IDs')
    pair = np.mean((values[:, None]-values[None])**2, axis=-1)
    np.fill_diagonal(pair, np.inf)
    neighbors = min(5, n-1)
    density = np.sort(pair, axis=1)[:, :neighbors].mean(1)
    density /= max(float(np.median(density)), 1e-8)
    mode_cost = np.mean((centers[:, None]-values[None])**2, axis=-1)
    mode_cost /= np.maximum(np.median(mode_cost, axis=1, keepdims=True), 1e-8)
    usage = np.zeros((modes, n), dtype=np.int64)
    expected = len(source_ids)/n
    order = sorted(range(len(source_ids)), key=lambda i:
                   hashlib.sha256(('1|'+source_ids[i]).encode()).hexdigest())
    selected = {}
    for index in order:
        paired = paired_indices[index]
        if not 0 <= paired < n:
            raise ValueError('Invalid paired index')
        cost = mode_cost + support_weight*density[None] + reuse_weight*usage/expected
        cost[:, paired] = np.inf
        rows, columns = linear_sum_assignment(cost)
        chosen = columns[np.argsort(rows)]
        assert len(set(chosen.tolist()+[paired])) == modes+1
        selected[source_ids[index]] = [paired, *chosen.tolist()]
        usage[np.arange(modes), chosen] += 1
    return selected, {'mean_selected_density_ratio': float(np.sum(usage*density[None])/usage.sum()),
                      'max_target_uses_per_slot': int(usage.max()),
                      'slot_unique_targets': (usage > 0).sum(1).tolist()}


def target_usage(mapping):
    result = []
    for rank in range(10):
        counts = Counter(v[rank] for v in mapping.values())
        p = np.array(list(counts.values()), dtype=float)/len(mapping)
        result.append({'rank': rank, 'distinct_GT': len(counts),
                       'effective_GT': float(np.exp(-np.sum(p*np.log(p)))),
                       'largest_fraction': float(p.max())})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    source = json.loads(args.source.read_text())
    frozen = json.loads((args.frozen_root/'pilot_provenance.json').read_text())
    if sha256_file(args.source) != frozen['frozen_target_manifest_source_sha256']:
        raise ValueError('Global prototypes differ from frozen TRAIN manifest')
    mean, std, centers = [np.asarray(source[k], dtype=np.float32) for k in
                          ('descriptor_mean', 'descriptor_std', 'prototype_centers')]
    reference = json.loads((args.frozen_root/'manifests/T0/rank_00.json').read_text())['mapping']
    facial = args.data_root/'train/facial-attributes'
    groups = defaultdict(list)
    for identifier in reference:
        role, session, name = identifier.split('/')
        groups[(role, session)].append(identifier)
    mapping, stats, target_div = {}, [], []
    file_hash = hashlib.sha256()
    for (role, session), identifiers in sorted(groups.items()):
        opposite = 'listener' if role == 'speaker' else 'speaker'
        paths = sorted((facial/opposite/session).glob('*.npy'))
        names = [x.relative_to(facial).as_posix() for x in paths]
        lookup = {name: i for i, name in enumerate(names)}
        raw = [_load_raw(x) for x in paths]
        for path in paths:
            file_hash.update(path.relative_to(facial).as_posix().encode())
            file_hash.update(path.read_bytes())
        values = (np.stack([full_reaction_descriptor(x).numpy() for x in raw])-mean)/std
        paired = [lookup[reference[x][0]] for x in identifiers]
        selected, detail = supported_assignment(values, centers, paired, identifiers)
        distance = official_distance_matrix(torch.stack([_relative_resample(x, 750) for x in raw]))
        for key, indices in selected.items():
            mapping[key] = [names[i] for i in indices]
            target_div.append(frdiv(distance, indices))
        stats.append({'source_role': role, 'session': session, **detail})
        print(f'T2R manifest {role}/{session}: {len(selected)} contexts', flush=True)
    assert set(mapping) == set(reference)
    assert all(mapping[k][0] == reference[k][0] for k in mapping)
    common = {'schema_version': 1, 'split': 'train', 'k': 10,
              'assignment_seed': 1, 'assignment_method': 'T2R_supported_prototypes_reuse',
              'preprocessing_version': 'supported-fixed-teacher-v1',
              'preserve_native_pair': True, 'alignment_policy': 'relative_time_masked',
              'fixed_pair_alignment_policy': 'relative_time_masked',
              'contexts': len(mapping), 'mapping': mapping,
              'support_weight': .5, 'reuse_weight': .5, 'nearest_neighbors': 5,
              'source_manifest_file_sha256': sha256_file(args.source)}
    for rank in range(10):
        manifest = {**common, 'rank': rank}
        manifest['manifest_hash'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (args.output_dir/f'rank_{rank:02d}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    usage = {'T2R': target_usage(mapping)}
    for arm in ('T0', 'T1'):
        original = json.loads((args.frozen_root/f'manifests/{arm}/rank_00.json').read_text())['mapping']
        usage[arm] = target_usage(original)
    summary = {'arm': 'T2R', 'contexts': len(mapping), 'TRAIN_target_FRDiv': float(np.mean(target_div)),
               'target_usage': usage, 'groups': stats, 'TRAIN_files_sha256': file_hash.hexdigest(),
               'prototype_source_sha256': sha256_file(args.source),
               'source_sha256': sha256_file(Path(__file__)),
               'policy': 'same nine global modes; kNN support cost plus per-slot reuse penalty; fixed pairing',
               'limitations': ['support/reuse combined policy; component effects not isolated',
                               'GT density is not a source-compatibility label',
                               'soft reuse penalty does not guarantee balanced usage']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('arm', 'contexts', 'TRAIN_target_FRDiv', 'target_usage')}, indent=2))


if __name__ == '__main__':
    main()
