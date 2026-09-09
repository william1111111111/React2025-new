"""Exact data provenance, stratified evaluation selection, random-crop preflight."""
import hashlib
import json
from collections import defaultdict
import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def feature_paths(dataset, index):
    emotion = dataset.records[index]
    relative = emotion.relative_to(dataset.directory / 'facial-attributes')
    return [dataset.directory/'audio-features'/relative, emotion,
            dataset.directory/'coefficients'/relative,
            dataset.directory/'facial-attributes/listener'/relative.parts[1]/relative.name]


def stratified_indices(dataset, per_session=4, seed=1501):
    """Fixed evaluation selection only; not a group training sampler."""
    groups = defaultdict(list)
    for i, path in enumerate(dataset.records):
        groups[path.parent.name].append(i)
    rng = torch.Generator().manual_seed(seed)
    selected = []
    for session in sorted(groups):
        pool = groups[session]
        if len(pool) < per_session:
            raise ValueError(f'{session} has fewer than {per_session} records')
        order = torch.randperm(len(pool), generator=rng)[:per_session].tolist()
        selected.extend(pool[j] for j in order)
    return selected


def random_crop_risk(source_length, target_length, clip_length):
    """Exact probability over source-only randint(0, max(0,S-T)) starts."""
    if source_length < 1 or target_length < 0 or clip_length < 1:
        raise ValueError('invalid lengths')
    max_start = max(0, source_length-clip_length)
    invalid = max(0, max_start-target_length+1)
    return dict(max_start=max_start, possible_starts=max_start+1,
                zero_overlap_starts=invalid, probability=invalid/(max_start+1))


def audit_random_crops(dataset):
    risks, mismatches, shortest = [], 0, None
    for index, path in enumerate(dataset.records):
        lengths = [np.load(p, mmap_mode='r', allow_pickle=False).shape[0]
                   for p in feature_paths(dataset, index)]
        source = min(lengths[:3])
        risk = random_crop_risk(source, lengths[3], dataset.clip_length)
        mismatches += int(source != lengths[3])
        shortest = source if shortest is None else min(shortest, source)
        if risk['zero_overlap_starts']:
            risks.append(dict(clip_id=str(path.relative_to(dataset.directory)),
                              source_length=source, target_length=lengths[3], **risk))
    return dict(split=dataset.split, clip_length=dataset.clip_length,
                scanned_records=len(dataset), total_length_mismatches=mismatches,
                shortest_source=shortest, records_with_zero_overlap_risk=len(risks), risks=risks)


def data_provenance(root, selections):
    """Hash all four selected raw arrays and the two normalization files."""
    paths = set()
    for dataset, indices in selections:
        for index in indices:
            paths.update(feature_paths(dataset, index))
    records = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                    sha256=sha256_file(p)) for p in sorted(paths)]
    normalization = [dict(path=f'external/FaceVerse/{name}', sha256=sha256_file(root/'external/FaceVerse'/name))
                     for name in ('mean_face.npy', 'std_face.npy')]
    return dict(data_files=records, data_file_list_sha256=canonical_hash(records),
                normalization_files=normalization)


def verify_provenance(root, manifest):
    if canonical_hash(manifest['data_files']) != manifest['data_file_list_sha256']:
        raise ValueError('data file list hash mismatch')
    for item in manifest['data_files'] + manifest['normalization_files']:
        if sha256_file(root/item['path']) != item['sha256']:
            raise ValueError(f'data hash mismatch: {item["path"]}')
