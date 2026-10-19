"""TRAIN-only cross-fitted, set-preserving conditional teacher assignment.

This is a mechanism probe, not an inference selector. The original paired target
and unordered Max-FRDiv target set remain exactly unchanged. Only ranks 1..9 move.
Frozen global TRAIN descriptor statistics/prototypes are inherited from T1;
regression and input preprocessing are fitted excluding each held-out session.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch

from coreset_reactor.dataset import full_reaction_descriptor
from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.supported_teacher_manifest import target_usage


def assign_set(values, conditional_centers, global_centers, anchor_weight=.25):
    """Rows are slots; columns are the nine unchanged non-paired candidates."""
    if values.shape != conditional_centers.shape or values.shape != global_centers.shape:
        raise ValueError('Expected equal [slots, descriptor_dim] shapes')
    cost = ((conditional_centers[:, None] - values[None])**2).mean(-1)
    cost += anchor_weight * ((global_centers[:, None] - values[None])**2).mean(-1)
    if not np.isfinite(cost).all():
        raise ValueError('Nonfinite assignment cost')
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]


def crossfit_predictions(x, y, sessions, *, seed=1, folds=4, components=64, ridge=.1):
    """One separate output function per slot, jointly solved multioutput ridge.

Ridge minimizes mean squared residual + ridge * squared coefficient norm.
The global T1 descriptor normalization is deliberately NOT re-estimated here.
"""
    result = np.empty_like(y)
    null = np.empty_like(y)
    splits, audit = [], []
    for train, test in GroupKFold(n_splits=folds).split(x, groups=sessions):
        assert not set(sessions[train]) & set(sessions[test])
        scaler = StandardScaler().fit(x[train])
        pca = PCA(n_components=min(components, len(train)-1, x.shape[1]),
                  whiten=True, svd_solver='randomized', random_state=seed)
        z = pca.fit_transform(scaler.transform(x[train]))
        model = Ridge(alpha=ridge*len(train), solver='cholesky')
        model.fit(z, y[train].reshape(len(train), -1))
        result[test] = model.predict(pca.transform(scaler.transform(x[test]))).reshape(y[test].shape)
        null[test] = y[train].mean(0)
        splits.append((train, test))
        audit.append({'fit_sessions': sorted(set(sessions[train].tolist())),
                      'heldout_sessions': sorted(set(sessions[test].tolist())),
                      'fit_count': len(train), 'heldout_count': len(test),
                      'conditional_mse': float(np.mean((result[test]-y[test])**2)),
                      'constant_slot_mse': float(np.mean((null[test]-y[test])**2))})
    return result, null, splits, audit


def shuffled_condition_indices(ids, seed=1):
    """Deterministic derangement within source role/session (same heldout fold)."""
    groups = defaultdict(list)
    for i, identifier in enumerate(ids):
        groups[tuple(identifier.split('/')[:2])].append(i)
    index = np.arange(len(ids))
    for group in groups.values():
        if len(group) < 2:
            raise ValueError('Cannot derange a singleton group')
        order = sorted(group, key=lambda i: hashlib.sha256(f'{seed}|{ids[i]}'.encode()).hexdigest())
        index[order] = np.roll(order, 1)
    assert (index != np.arange(len(ids))).all()
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    original_path = args.frozen_root/'manifests/T1/rank_00.json'
    original = json.loads(original_path.read_text())
    frozen = json.loads((args.frozen_root/'pilot_provenance.json').read_text())
    if sha256_file(args.source) != frozen['frozen_target_manifest_source_sha256']:
        raise ValueError('Not the frozen T1 descriptor definition')
    if sha256_file(original_path) != frozen['target_manifest_sha256']['T1/rank_00']:
        raise ValueError('T1 rank manifest changed')
    source = json.loads(args.source.read_text())
    mean, std, centers = [np.asarray(source[k], dtype=np.float32) for k in
                         ('descriptor_mean', 'descriptor_std', 'prototype_centers')]
    mapping = original['mapping']
    ids = sorted(mapping)
    sessions = np.asarray([s.split('/')[1] for s in ids])
    train_root = args.data_root/'train'
    # Pooled inputs never include opposite-role target features or source IDs.
    features, descriptors, hashes = [], {}, {}
    all_ids = sorted(set(ids) | {v for row in mapping.values() for v in row})
    for i, identifier in enumerate(all_ids):
        path = train_root/'facial-attributes'/identifier
        raw = np.load(path, allow_pickle=False).astype(np.float32)
        descriptors[identifier] = (full_reaction_descriptor(torch.from_numpy(raw)).numpy()-mean)/std
        hashes[str(path.relative_to(train_root))] = sha256_file(path)
        if i % 500 == 0:
            print(f'Descriptors {i}/{len(all_ids)}', flush=True)
    for i, identifier in enumerate(ids):
        row = []
        for directory, width in (('audio-features', 768), ('facial-attributes', 25), ('coefficients', 58)):
            path = train_root/directory/identifier
            value = np.load(path, allow_pickle=False).astype(np.float32)
            if value.ndim == 3 and value.shape[1] == 1:
                value = value[:, 0]
            if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
                raise ValueError(f'Invalid source feature: {path}')
            row.extend([value.mean(0), value.std(0)])
            hashes[str(path.relative_to(train_root))] = sha256_file(path)
        features.append(np.concatenate(row))
        if i % 500 == 0:
            print(f'Source features {i}/{len(ids)}', flush=True)
    x = np.stack(features)
    y = np.stack([np.stack([descriptors[t] for t in mapping[k][1:]]) for k in ids])
    pred, null, splits, folds = crossfit_predictions(x, y, sessions)
    shuffle = shuffled_condition_indices(ids)
    np.savez_compressed(args.output_dir/'crossfit_predictions.npz', ids=np.asarray(ids),
                        conditional=pred, constant=null, shuffle_indices=shuffle)
    arms = {}
    # Constant-condition is a no-source learned-prototype diagnostic, not trained
    # in this initial two-arm probe; its manifest is retained for a later control.
    for arm, predictions in (('conditional', pred), ('shuffled', pred[shuffle]), ('constant', null)):
        assigned, changes, gains = {}, [], []
        for i, identifier in enumerate(ids):
            cols = assign_set(y[i], predictions[i], centers)
            old = mapping[identifier]
            assigned[identifier] = [old[0]] + [old[j+1] for j in cols]
            assert assigned[identifier][0] == old[0]
            assert sorted(assigned[identifier]) == sorted(old)
            assert len(set(old)) == 10
            changes.append(float(np.mean(cols != np.arange(9))))
            gains.append(float(np.mean((y[i]-predictions[i])**2) -
                               np.mean((y[i][cols]-predictions[i])**2)))
        out = args.output_dir/'manifests'/arm
        out.mkdir(parents=True)
        for rank in range(10):
            manifest = {**original, 'rank': rank, 'mapping': assigned,
                        'assignment_method': f'T1_set_preserving_{arm}_crossfit',
                        'preprocessing_version': 'conditional-slot-v1',
                        'source_t1_manifest_sha256': sha256_file(original_path)}
            manifest.pop('manifest_hash', None)
            manifest['manifest_hash'] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                                        separators=(',', ':')).encode()).hexdigest()
            (out/f'rank_{rank:02d}.json').write_text(json.dumps(manifest, indent=2)+'\n')
        arms[arm] = {'fraction_nonanchor_slots_changed': float(np.mean(changes)),
                     'fraction_contexts_changed': float(np.mean(np.asarray(changes)>0)),
                     'conditional_assignment_cost_reduction': float(np.mean(gains)),
                     'target_usage': target_usage(assigned), 'all_target_sets_identical': True,
                     'paired_targets_identical': True,
                     'manifest_hashes': {str(r): sha256_file(out/f'rank_{r:02d}.json') for r in range(10)}}
    pred_mse = float(np.mean((pred-y)**2))
    null_mse = float(np.mean((null-y)**2))
    summary = {'experiment': 'set-preserving TRAIN cross-fitted conditional slot assignment',
               'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'script_sha256': sha256_file(Path(__file__)),
               'source_manifest_sha256': sha256_file(args.source),
               'frozen_rank00_sha256': sha256_file(original_path),
               'contexts': len(ids), 'sessions': sorted(set(sessions.tolist())),
               'protocol': {'split': 'train', 'seed': 1, 'folds': 4, 'group': 'session (both roles)',
                            'features': 'full-source mean/std audio768+emotion25+3DMM58',
                            'pca': 64, 'ridge_mean_loss_weight': .1, 'global_mode_anchor_weight': .25,
                            'descriptor': 'frozen T1 global TRAIN 602D statistics; inherited, not fold-fitted',
                            'validation_used': False, 'inference_assignment': False},
               'folds': folds, 'oof_mse': pred_mse, 'constant_mse': null_mse,
               'shuffled_mse': float(np.mean((pred[shuffle]-y)**2)),
               'oof_relative_mse_gain': 1-pred_mse/null_mse,
               'arms': arms, 'seconds': time.monotonic()-started}
    (args.output_dir/'input_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('arms', 'folds')}, indent=2), flush=True)
    print(json.dumps({k: {a:b for a,b in v.items() if a not in ('target_usage','manifest_hashes')}
                      for k,v in arms.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
