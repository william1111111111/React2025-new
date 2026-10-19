"""Read-only TRAIN target-search/prior audit and full-VAL diversity decomposition.

Search is a deterministic multistart feasible lower bound, not a certified
ceiling. Candidate sets are diagnostics, never installed as training targets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from coreset_reactor.dataset_gt_oracle import official_distance_matrix, frdiv
from coreset_reactor.teacher_diversity_decomposition import decompose
from coreset_reactor.train_mode_reference_teacher import entropy_matched_log_prior
from regnn.conditional_data import relative_time_resample_reaction


def improve(distance, initial, anchor):
    """Vectorized deterministic 1-swap ascent with an immutable paired item."""
    selected = list(initial)
    if len(selected) != len(set(selected)) or anchor not in selected:
        raise ValueError('Need unique candidates including anchor')
    if selected[0] != anchor:
        selected.remove(anchor)
        selected.insert(0, anchor)
    n = len(distance)
    for _ in range(100):
        outside = np.setdiff1d(np.arange(n), selected)
        if not len(outside):
            break
        sums = distance[:, selected].sum(1)
        outgoing = np.array(selected[1:])
        gains = sums[outside, None] - distance[np.ix_(outside, outgoing)] - sums[outgoing][None]
        flat = int(gains.argmax())
        row, column = np.unravel_index(flat, gains.shape)
        if gains[row, column] <= 1e-10:
            break
        selected[column + 1] = int(outside[row])
    return selected


def search(distance, original, seed, starts=16):
    anchor = original[0]
    best = improve(distance, original, anchor)
    best_value = frdiv(distance, best)
    greedy = [anchor]
    while len(greedy) < len(original):
        gains = distance[:, greedy].sum(1)
        gains[greedy] = -np.inf
        greedy.append(int(gains.argmax()))
    rng = np.random.default_rng(seed)
    others = np.setdiff1d(np.arange(len(distance)), [anchor])
    initializations = [greedy] + [[anchor, *rng.choice(others, len(original)-1, replace=False).tolist()]
                                for _ in range(starts)]
    values = []
    for initial in initializations:
        chosen = improve(distance, initial, anchor)
        value = frdiv(distance, chosen)
        values.append(value)
        if value > best_value + 1e-12:
            best, best_value = chosen, value
    return best, best_value, max(values) - min(values)


def paired_random_expectation(distance, anchor, k=10):
    n = len(distance)
    if n < k or k < 2:
        raise ValueError('Invalid subset size')
    other = np.setdiff1d(np.arange(n), [anchor])
    anchor_term = (k-1)/(n-1) * distance[anchor, other].sum()
    others_term = ((k-1)*(k-2)/((n-1)*(n-2)) *
                   distance[np.ix_(other, other)].sum()/2) if n > 2 else 0
    return float((anchor_term + others_term)/(k*(k-1)/2))


def summarize(rows):
    return {key: {'mean':float(np.mean([r[key] for r in rows])),
                  'p10':float(np.quantile([r[key] for r in rows], .1)),
                  'median':float(np.median([r[key] for r in rows])),
                  'p90':float(np.quantile([r[key] for r in rows], .9))}
            for key, value in rows[0].items() if isinstance(value, (int,float))}


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def audit_train(data_root, manifest_path, output, search_targets, starts):
    mapping = json.loads(manifest_path.read_text())['mapping']
    facial = data_root/'train/facial-attributes'
    groups = defaultdict(list)
    for path in sorted(facial.glob('*/*/*.npy')):
        groups['/'.join(path.relative_to(facial).parts[:2])].append(path)
    scale_values = []
    for paths in groups.values():
        for p in paths:
            value = np.load(p).astype(np.float32)
            if value.ndim != 2 or value.shape[1] != 25 or not len(value):
                raise ValueError('Invalid TRAIN trajectory')
            if search_targets and value.shape != (750,25):
                raise ValueError('This diagnostic requires frozen 750-frame converted TRAIN clips')
            scale_values.append(np.concatenate((value.mean(0), value.std(0))))
    scale = np.maximum(np.stack(scale_values).std(0), .03)
    sources = defaultdict(list)
    for source, targets in mapping.items():
        sources['/'.join(targets[0].split('/')[:2])].append((source,targets))
    rows, selections = [], {}
    for group_index, (group, entries) in enumerate(sorted(sources.items())):
        paths = groups[group]
        ids = {p.relative_to(facial).as_posix():i for i,p in enumerate(paths)}
        raw = torch.stack([relative_time_resample_reaction(
            torch.from_numpy(np.load(p).astype(np.float32)),750) for p in paths])
        official = raw.clone()
        official[..., :15] = official[..., :15].round()
        raw_distance = official_distance_matrix(raw)
        distance = official_distance_matrix(official)
        desc = torch.cat((raw.mean(1), raw.std(1,unbiased=False)), -1).numpy()/scale
        cost = np.square(desc[:,None]-desc[None]).mean(-1)
        priors = {count: np.exp(entropy_matched_log_prior(cost, count)) for count in (2,4,8)}
        for source, targets in entries:
            selected = [ids[t] for t in targets]
            if len(set(selected)) != 10:
                raise ValueError('Fixed GT must be ten distinct trajectories')
            seed = int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], 'big')
            original = frdiv(distance, selected)
            row = {'source':source, 'session':group, 'pool_size':len(paths),
                   'fixed_raw_FRDiv':frdiv(raw_distance, selected),
                   'fixed_rounded_FRDiv':original,
                   'random_paired9_expectation':paired_random_expectation(distance, selected[0]),
                   'raw_AU_nonbinary_fraction':float(((raw[..., :15] != 0)&(raw[..., :15] != 1)).float().mean()),
                   'effective8_pool_fraction':min(8,len(paths))/len(paths)}
            for count, prior in priors.items():
                p = prior[selected[1:]]
                self_mass = p[np.arange(9), selected[1:]]
                root = np.sqrt(p)
                overlap = root @ root.T
                row[f'prior{count}_fixed_mass'] = float(self_mass.mean())
                row[f'prior{count}_expected_descriptor_displacement'] = float((p*cost[selected[1:]]).sum(1).mean())
                row[f'prior{count}_slot_overlap'] = float(overlap[np.triu_indices(9,1)].mean())
            if search_targets:
                best, value, spread = search(distance, selected, seed, starts)
                row.update({'searched_FRDiv':value, 'search_gain':value-original,
                            'search_relative_gain':value/max(original,1e-12)-1,
                            'multistart_score_spread':spread,
                            'selected_retained_fraction':len(set(best)&set(selected))/10})
                selections[source] = [paths[i].relative_to(facial).as_posix() for i in best]
            rows.append(row)
        if (group_index+1)%10 == 0:
            print(f'{output.name}: groups {group_index+1}/{len(sources)}, contexts {len(rows)}',flush=True)
    result = {'split':'TRAIN only', 'contexts':len(rows), 'groups':len(sources),
              'fixed_target_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'representation':'same 750-frame relative alignment and AU rounding for every compared set',
              'cross_dataset_prior_note':'Prior-only cross-dataset comparison standardizes all references to 750 frames; does not claim exact priors for short-source runtime contexts.',
              'search':'original + greedy + random starts, paired-preserving 1-swap; lower bound, NOT proven optimum',
              'random_starts':starts if search_targets else 0,
              'prior_note':'Entropy count alone is not semantic radius. Overlap/displacement are TRAIN statistics, not causal proof.',
              'summary':summarize(rows), 'rows':rows}
    if selections:
        write(output.with_name('diagnostic_candidate_sets.json'), {
            'purpose':'read-only diagnostic, NOT installed in training', 'split':'train',
            'mapping':selections})
    write(output, result)
    return result


def val_decomposition(root, output):
    labels = ('old_teacher','new_teacher','old_contrast','new_basic','new_contrast')
    summary, rows = {}, {}
    expected_ids = None
    for label in labels:
        path = root/'eval'/f'{label}_full.json'
        ledger = json.loads(path.read_text())
        bank = torch.load(path.with_suffix('.predictions.pt'), map_location='cpu')
        ids = bank['clip_ids']
        assert len(ids) == 1124 and ids == [r['clip_id'] for r in ledger['rows']]
        assert bank['checkpoint_sha256'] == ledger['checkpoint_sha256']
        if expected_ids is None: expected_ids = ids
        assert ids == expected_ids
        measures = {'rounded':[], 'soft':[]}
        for kind,key in (('rounded','official_predictions'),('soft','soft_predictions')):
            for p in bank[key]:
                values = decompose(p)
                if kind == 'soft':
                    au = p[..., :15]
                    values['AU_near_rounding_boundary_fraction'] = float(((au>.4)&(au<.6)).float().mean())
                measures[kind].append(values)
        means = {kind:{k:float(np.mean([r[k] for r in rr])) for k in rr[0]}
                 for kind,rr in measures.items()}
        np.testing.assert_allclose(means['rounded']['FRDiv'], ledger['metrics']['FRDiv'], atol=1e-6)
        np.testing.assert_allclose(means['soft']['FRDiv'], ledger['metrics']['soft_FRDiv'], atol=1e-6)
        summary[label] = {'metrics':ledger['metrics'], **means}
        rows[label] = measures
        del bank
        print(f'full VAL1124 decomposition: {label} complete',flush=True)
    result = {'contexts':1124, 'protocol':'fixed full-VAL predictions, exact orthogonal FRDiv decomposition',
              'summary':summary, 'clip_ids':expected_ids, 'rows':rows}
    write(output, result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data2025',type=Path,required=True)
    parser.add_argument('--manifest2025',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=16)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started=time.monotonic()
    provenance=json.loads((args.run_root/'provenance.json').read_text())
    val=val_decomposition(args.run_root,args.output_dir/'val_decomposition.json')
    train=audit_train(Path(provenance['data_root']), args.run_root/'frozen_T1/rank_00/manifest.json',
                      args.output_dir/'train2024.json',True,args.starts)
    # Standardized 750-frame prior comparison. Raw full-trajectory statistics
    # define the scale exactly as in training, before reference resampling.
    try:
        cross=audit_train(args.data2025,args.manifest2025,args.output_dir/'train2025_prior.json',False,0)
        cross_summary=cross['summary']
    except ValueError as error:
        cross_summary={'not_computed':str(error)}
    summary={'finished':True,'train2024':train['summary'],'val':val['summary'],
             'train2025_prior':cross_summary, 'elapsed_seconds':time.monotonic()-started,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'caveats':['TRAIN target FRDiv and VAL output FRDiv are separate populations.',
                        'No checkpoints, targets, or losses were changed.',
                        'Multistart search is not a certified GT ceiling.']}
    write(args.output_dir/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('val',)},indent=2),flush=True)


if __name__=='__main__': main()
