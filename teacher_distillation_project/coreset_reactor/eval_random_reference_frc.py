"""Sampling-only FRC ablation on frozen predictions and postprocessed VAL GT.

Paired+9 follows the baseline reference-set policy, not its full TEST pipeline.
Uniform10 is reported separately. Same draws across models; no GT at inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from coreset_reactor.fast_frc import pairwise_frc


def draw_indices(count, clip_id, seed, policy):
    if count < 1:
        raise ValueError('Empty GT pool')
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f'{seed}:{clip_id}'.encode()).digest()[:8], 'big'))
    if policy == 'paired_plus9':
        others = list(range(1, count)) or [0]
        return [0] + (rng.sample(others, 9) if len(others) >= 9 else rng.choices(others, k=9))
    if policy == 'uniform10':
        pool = list(range(count))
        return rng.sample(pool, 10) if count >= 10 else rng.choices(pool, k=10)
    raise ValueError(policy)


def stats(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)),
            'minimum': float(np.min(values)), 'maximum': float(np.max(values)),
            'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', type=Path, required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=100)
    parser.add_argument('--first-seed', type=int, default=1234)
    parser.add_argument('--expected-contexts', type=int, default=571)
    args = parser.parse_args()
    if len(args.names) != len(args.evaluations) or len(set(args.names)) != len(args.names):
        raise ValueError('Need distinct names, one per evaluation')
    if args.draws < 2:
        raise ValueError('Need >=2 draws')
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ledgers = [json.loads(p.read_text()) for p in args.evaluations]
    cached = [torch.load(p.with_suffix('.predictions.pt'), map_location='cpu') for p in args.evaluations]
    reference = ledgers[0]
    contexts = len(reference['rows'])
    if contexts != args.expected_contexts:
        raise ValueError(f'Expected full VAL{args.expected_contexts}, got {contexts}')
    for ledger, cache in zip(ledgers, cached):
        assert cache['checkpoint_sha256'] == ledger['checkpoint_sha256']
        assert cache['clip_ids'] == [r['clip_id'] for r in ledger['rows']]
        for key in ('clip_id', 'target_cache_key', 'gt_count'):
            assert [r[key] for r in ledger['rows']] == [r[key] for r in reference['rows']]
    seeds = list(range(args.first_seed, args.first_seed + args.draws))
    policies = ('paired_plus9', 'uniform10')
    scores = np.zeros((len(ledgers), len(policies), contexts, args.draws))
    all_scores = np.zeros((len(ledgers), contexts))
    rows = []
    for index, row in enumerate(reference['rows']):
        payload = torch.load(args.target_cache / (row['target_cache_key'] + '.pt'), map_location='cpu')
        assert payload['fingerprint'] == row['target_cache_key']
        targets = payload['targets'].numpy()
        assert len(targets) == row['gt_count']
        selections = np.asarray([[draw_indices(len(targets), row['clip_id'], seed, policy)
                                  for seed in seeds] for policy in policies])
        predictions = np.concatenate([c['official_predictions'][index].numpy() for c in cached])
        matrix = pairwise_frc(targets, predictions).reshape(len(ledgers), 10, len(targets))
        for model, ledger in enumerate(ledgers):
            all_scores[model, index] = matrix[model].max(-1).sum()
            np.testing.assert_allclose(all_scores[model, index], sum(ledger['rows'][index]['frc']),
                                       atol=1e-6, rtol=0)
            for policy_index in range(len(policies)):
                # [ten predictions, draws, ten sampled GT]
                scores[model, policy_index, index] = matrix[model][:, selections[policy_index]].max(-1).sum(0)
                assert np.all(scores[model, policy_index, index] <= all_scores[model, index] + 1e-8)
        np.savez_compressed(args.output_dir / f'context_{index:04d}.npz',
                            pairwise_frc=matrix, selected_indices=selections)
        rows.append({'clip_id': row['clip_id'], 'target_cache_key': row['target_cache_key'],
                     'gt_count': len(targets)})
        if (index + 1) % 25 == 0:
            (args.output_dir / 'progress.json').write_text(json.dumps({'completed': index + 1, 'total': contexts}))
            print(f'{index + 1}/{contexts}', flush=True)
    summary = {'finished': True, 'contexts': contexts, 'draws': args.draws, 'seeds': seeds,
               'protocol': 'Frozen predictions + fixed processed GT; sampling-only ablation. Not full official TEST reproduction.',
               'aggregation': f'sum best mean-channel CCC over ten predictions, then mean over {contexts} contexts',
               'randomness': 'Same per-context seeded draws across models; std is sampling variability, not training uncertainty',
               'FRD': 'not recomputed', 'models': {},
               'elapsed_seconds': time.monotonic() - started,
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(__file__).with_name('fast_frc.py'))}}
    report = ['# Frozen teacher/student random-reference FRC', '',
              f'Full VAL{contexts}. {args.draws} draws, fixed predictions and processed GT.',
              'Sampling-only comparison, not official TEST reproduction. FRD not recomputed.', '',
              '| Model | All-session FRC | Paired+9 mean ± std | Loss | Uniform10 mean ± std | Loss | Unchanged FRDiv |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for model, name in enumerate(args.names):
        baseline = float(all_scores[model].mean())
        np.testing.assert_allclose(baseline, ledgers[model]['metrics']['FRC'], atol=1e-6, rtol=0)
        result = {'all_session_FRC': baseline, 'unchanged_FRDiv': ledgers[model]['metrics']['FRDiv'],
                  'checkpoint_sha256': ledgers[model]['checkpoint_sha256'], 'policies': {}}
        for p, policy in enumerate(policies):
            values = scores[model, p]
            s = stats(values.mean(0))
            s.update({'absolute_loss': baseline - s['mean'],
                      'relative_loss_percent': 100 * (baseline - s['mean']) / baseline,
                      'per_draw_mean': values.mean(0).tolist(),
                      'context_sampling_std_p50_p90_max': np.quantile(values.std(1, ddof=1), [.5, .9, 1]).tolist()})
            result['policies'][policy] = s
        summary['models'][name] = result
        a, b = [result['policies'][p] for p in policies]
        report.append(f"| {name} | {baseline:.6f} | {a['mean']:.6f} ± {a['std']:.6f} | "
                      f"{a['relative_loss_percent']:.2f}% | {b['mean']:.6f} ± {b['std']:.6f} | "
                      f"{b['relative_loss_percent']:.2f}% | {result['unchanged_FRDiv']:.6f} |")
    np.savez_compressed(args.output_dir / 'scores.npz', scores=scores, all_scores=all_scores,
                        models=args.names, policies=policies, seeds=seeds)
    (args.output_dir / 'rows.json').write_text(json.dumps(rows))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output_dir / 'RANDOM_REFERENCE_FRC.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
