"""Generate, score and summarize frozen common-VA correction diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.common_va import (VARIANTS, feasible_shift, apply_shift,
    optimize_frc_shift, SharedVAQualityPool, strict_choice, ccc_channels)
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.paired_data import PairedReactionDataset

ROOT = Path(__file__).resolve().parents[1]


def load_gt(cache, row):
    key = row['target_cache_key']
    state = torch.load(cache/f'{key}.pt', map_location='cpu')
    if state['fingerprint'] != key:
        raise ValueError('Frozen target cache mismatch')
    return state['targets']


def generate(args):
    args.root.mkdir(parents=True, exist_ok=False)
    (args.root/'proposals').mkdir()
    baseline = json.loads(args.baseline.read_text())
    quality = json.loads(args.quality.read_text())
    predictions = torch.load(args.baseline.with_suffix('.predictions.pt'), map_location='cpu')
    quality_predictions = torch.load(args.quality.with_suffix('.predictions.pt'), map_location='cpu')
    assert predictions['clip_ids'] == quality_predictions['clip_ids']
    assert [(r['clip_id'], r['target_cache_key']) for r in baseline['rows']] == [(r['clip_id'], r['target_cache_key']) for r in quality['rows']]
    count = min(args.limit or len(baseline['rows']), len(baseline['rows']))
    pilot = sorted(np.random.default_rng(1234).choice(count, min(64, count), replace=False).tolist())
    files = ['coreset_reactor/common_va.py', 'coreset_reactor/common_va_experiment.py',
             'coreset_reactor/run_common_va.py', 'coreset_reactor/fast_frc.py',
             'coreset_reactor/oracle_select.py', 'mam_reactor/tools/compute_frd_resumable.py',
             'coreset_reactor/results/COMMON_VA_PROTOCOL.md']
    provenance = {'base_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': {name: sha256_file(ROOT/name) for name in files},
                  'baseline': str(args.baseline.resolve()), 'quality': str(args.quality.resolve()),
                  'target_cache': str(args.target_cache.resolve()), 'data_root': str(args.data_root.resolve()),
                  'asset_mam_root': str(args.asset_mam_root.resolve()),
                  'baseline_sha256': sha256_file(args.baseline), 'quality_sha256': sha256_file(args.quality),
                  'baseline_prediction_sha256': sha256_file(args.baseline.with_suffix('.predictions.pt')),
                  'quality_prediction_sha256': sha256_file(args.quality.with_suffix('.predictions.pt')),
                  'contexts': count, 'pilot_indices': pilot, 'eval_seed': 1234,
                  'oracle': {'steps': 100, 'controls_per_channel': 16, 'learning_rate': .03,
                             'initialization': 'zeros', 'scales': [.25, .5, 1.0]},
                  'variants': VARIANTS, 'teacher_training_updates': 0,
                  'q0': 'weights unchanged; output VA receives the same shift',
                  'quality_claim': 'per-context zero-slack FRC/FRD safety; local finite-family oracle only'}
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    torch.cuda.set_device(torch.device(args.device))
    started = time.monotonic()
    for index in range(count):
        row = baseline['rows'][index]
        pred = predictions['official_predictions'][index]
        other = quality_predictions['official_predictions'][index]
        target = load_gt(args.target_cache, row)
        assert pred.shape == other.shape == (10, target.shape[1], 25)
        frc0 = candidate_frc(target.numpy(), pred.numpy())
        np.testing.assert_allclose(frc0, row['frc'], atol=1e-7, rtol=1e-6)
        if index == 0:
            torch_frc = ccc_channels(target.double(), pred.double()).mean(-1).max(-1).values.numpy()
            np.testing.assert_allclose(torch_frc, frc0, atol=1e-6, rtol=1e-5)
        tick = time.monotonic()
        shared = feasible_shift(pred, other[..., 15:17].mean(0)-pred[..., 15:17].mean(0))
        gt_free_seconds = time.monotonic()-tick
        tick = time.monotonic()
        optimized, objective = optimize_frc_shift(pred, target, args.device)
        oracle_seconds = time.monotonic()-tick
        shifts = torch.stack([torch.zeros_like(shared), shared, .25*optimized, .5*optimized, optimized])
        variants = [apply_shift(pred, x) for x in shifts]
        frc = candidate_frc(target.numpy(), torch.cat(variants).numpy()).reshape(len(VARIANTS), 10)
        drift = max(float(((x.double()-x.double().mean(0))-(pred.double()-pred.double().mean(0))).abs().max()) for x in variants)
        assert drift < 3e-7
        assert all(float(x[..., 15:17].abs().max()) <= 1+1e-7 for x in variants)
        torch.save({'index': index, 'clip_id': row['clip_id'], 'target_cache_key': row['target_cache_key'],
                    'shifts': shifts, 'frc': frc, 'geometry_max_error': drift,
                    'gt_free_seconds': gt_free_seconds, 'oracle_seconds': oracle_seconds,
                    'oracle_objective': objective}, args.root/f'proposals/{index:06d}.pt')
        if (index+1) % 25 == 0 or index+1 == count:
            progress = {'completed': index+1, 'total': count, 'elapsed_seconds': time.monotonic()-started}
            (args.root/'generation.progress.json').write_text(json.dumps(progress)+'\n')
            print(json.dumps(progress), flush=True)
    provenance['generation_seconds'] = time.monotonic()-started
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def score(args):
    from framework.metrics import compute_s_mse, compute_FRVar
    provenance = json.loads((args.root/'provenance.json').read_text())
    baseline_path = Path(provenance['baseline'])
    baseline = json.loads(baseline_path.read_text())
    predictions = torch.load(baseline_path.with_suffix('.predictions.pt'), map_location='cpu')
    cache = Path(provenance['target_cache'])
    data = PairedReactionDataset(Path(provenance['data_root']), 'val', 750, crop_mode='center',
                                normalization_dir=Path(provenance['asset_mam_root'])/'external/FaceVerse')
    facial = Path(provenance['data_root'])/'val/facial-attributes'
    pools = {}
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools.setdefault(path.parent.name, []).append(path)
    all_indices = list(range(provenance['contexts']))
    if args.phase == 'pilot':
        chosen = provenance['pilot_indices']
    else:
        chosen = [i for i in all_indices if i not in set(provenance['pilot_indices'])]
    chosen = chosen[args.shard_index::args.shard_count]
    ledger = args.root/'rows'
    ledger.mkdir(exist_ok=True)
    started = time.monotonic()
    pool = SharedVAQualityPool(workers=args.workers, max_targets=max(r['gt_count'] for r in baseline['rows']))
    try:
        for completed, index in enumerate(chosen, 1):
            output = ledger/f'{index:06d}.json'
            if output.exists():
                raise FileExistsError(output)
            proposal = torch.load(args.root/f'proposals/{index:06d}.pt', map_location='cpu')
            original = baseline['rows'][index]
            assert proposal['clip_id'] == original['clip_id'] == predictions['clip_ids'][index]
            pred = predictions['official_predictions'][index]
            target = load_gt(cache, original)
            variants = [apply_shift(pred, shift) for shift in proposal['shifts']]
            tick = time.monotonic()
            frd, calls = pool.score_variants(variants, target)
            frd_seconds = time.monotonic()-tick
            np.testing.assert_allclose(frd[0].sum(), original['exact_FRD'], atol=1e-7, rtol=1e-8)
            frc = candidate_frc(target.numpy(), torch.cat(variants).numpy()).reshape(len(VARIANTS), 10)
            np.testing.assert_allclose(frc, proposal['frc'], atol=1e-7, rtol=1e-6)
            sample = data[index]
            assert sample['clip_id'] == original['clip_id']
            length, start = int(sample['source_lengths']), int(sample['crop_start'])
            paired = (facial/sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
            gt_paths = [paired, *[p for p in pools[sample['session_id']] if p != paired]]
            raw = [torch.from_numpy(np.load(p, allow_pickle=False).astype(np.float32)) for p in gt_paths]
            raw_target = aligned_targets(raw, length, start)
            raw_frc = candidate_frc(raw_target.numpy(), torch.cat(variants).numpy()).reshape(len(VARIANTS), 10).sum(1)
            metrics = []
            for j, value in enumerate(variants):
                assert torch.equal(value[..., :15], pred[..., :15]) and torch.equal(value[..., 17:], pred[..., 17:])
                metrics.append({'frc': frc[j].tolist(), 'FRC': float(frc[j].sum()),
                                'frd_per_rank': frd[j].tolist(), 'exact_FRD': float(frd[j].sum()),
                                'FRDiv': float(compute_s_mse([value])), 'FRVar': float(compute_FRVar([value])),
                                'raw_FRC': float(raw_frc[j])})
            assert max(abs(x['FRDiv']-metrics[0]['FRDiv']) for x in metrics) < 1e-6
            strict = strict_choice(frc.sum(1), frd.sum(1))
            unconstrained = int(np.argmax(frc.sum(1)))
            row = {'index': index, 'clip_id': proposal['clip_id'], 'target_cache_key': proposal['target_cache_key'],
                   'gt_count': len(target), 'variants': dict(zip(VARIANTS, metrics)),
                   'strict_choice': VARIANTS[strict], 'FRC_only_choice': VARIANTS[unconstrained],
                   'geometry_max_error': proposal['geometry_max_error'], 'dtw_group_calls': calls.tolist(),
                   'exact_FRD_seconds': frd_seconds, 'gt_free_seconds': proposal['gt_free_seconds'],
                   'oracle_seconds': proposal['oracle_seconds']}
            output.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
            if completed % 10 == 0 or completed == len(chosen):
                progress = {'completed': completed, 'total': len(chosen), 'elapsed_seconds': time.monotonic()-started}
                (args.root/f'{args.phase}_shard{args.shard_index}.progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    finally:
        pool.close()


def summarize(args):
    provenance = json.loads((args.root/'provenance.json').read_text())
    indices = provenance['pilot_indices'] if args.phase == 'pilot' else list(range(provenance['contexts']))
    rows = [json.loads((args.root/f'rows/{i:06d}.json').read_text()) for i in indices]
    keys = ['FRC', 'exact_FRD', 'FRDiv', 'FRVar', 'raw_FRC']
    summaries, selections = {}, {}
    for arm in [*VARIANTS, 'FRC_only_oracle', 'strict_oracle']:
        values = [r['variants'][r['FRC_only_choice'] if arm == 'FRC_only_oracle' else r['strict_choice'] if arm == 'strict_oracle' else arm] for r in rows]
        summaries[arm] = {k: float(np.mean([x[k] for x in values])) for k in keys}
        if arm.endswith('oracle'):
            field = 'strict_choice' if arm == 'strict_oracle' else 'FRC_only_choice'
            selections[arm] = dict(Counter(r[field] for r in rows))
    differences = {}
    for arm in ['quality_centroid', 'FRC_only_oracle', 'strict_oracle']:
        field = 'strict_choice' if arm == 'strict_oracle' else 'FRC_only_choice'
        delta = {k: np.array([r['variants'][arm if arm == 'quality_centroid' else r[field]][k]-r['variants']['baseline'][k] for r in rows]) for k in keys}
        differences[arm] = {}
        for key, values in delta.items():
            groups = {}
            for row, value in zip(rows, values):
                groups.setdefault(row['clip_id'].split('/')[1], []).append(value)
            sums, counts = np.array([sum(v) for v in groups.values()]), np.array([len(v) for v in groups.values()])
            draws = np.random.default_rng(1234).integers(0, len(groups), (5000, len(groups)))
            bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
            differences[arm][key] = {'mean': float(values.mean()), 'quantiles_P0_P10_P50_P90_P100': np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
                                    'improvement_fraction': float(np.mean(values < 0 if key == 'exact_FRD' else values > 0)),
                                    'session_bootstrap_CI95': np.quantile(bootstrap, [.025, .975]).tolist()}
    for row in rows:
        a, b = row['variants'][row['strict_choice']], row['variants']['baseline']
        assert a['FRC'] >= b['FRC'] and a['exact_FRD'] <= b['exact_FRD']
    session_summary = {}
    for session in sorted({r['clip_id'].split('/')[1] for r in rows}):
        subset = [r for r in rows if r['clip_id'].split('/')[1] == session]
        session_summary[session] = {'contexts': len(subset), **{
            arm: {k: float(np.mean([r['variants'][r['strict_choice'] if arm == 'strict_oracle' else arm][k] for r in subset])) for k in keys}
            for arm in ['baseline', 'quality_centroid', 'strict_oracle']}}
    summary = {'contexts': len(rows), 'metrics': summaries, 'selection_composition': selections,
               'paired_deltas': differences, 'per_session': session_summary,
               'geometry_max_error': max(r['geometry_max_error'] for r in rows),
               'strict_all_contexts_safe': True, 'source_sha256': sha256_file(Path(__file__)),
               'speed': {k: float(np.mean([r[k] for r in rows])) for k in ['gt_free_seconds', 'oracle_seconds', 'exact_FRD_seconds']},
               'dtw_group_calls': np.sum([r['dtw_group_calls'] for r in rows], 0).tolist(),
               'interpretation': 'local GT-using diagnostic, not a deployable model or proven maximum'}
    (args.root/f'{args.phase}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.report:
        lines = ['# Frozen common-VA correction: quality versus invariant diversity', '',
                 f"REACT2025; {len(rows)} VAL contexts; all-session opposite-role GT; same processed-GT cache; official AU rounding and exact rolling FRD. Not official random-10 GT sampling.", '',
                 '| Method | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ |', '|---|---:|---:|---:|---:|---:|']
        for arm, values in summaries.items():
            lines.append(f'| {arm} | '+' | '.join(f'{values[k]:.6f}' for k in keys)+' |')
        lines += ['', 'Only `quality_centroid` is GT-free, with fixed coefficient1. Every oracle arm uses VAL GT and is diagnostic only. Selection is over complete ten-prediction sets, never a mixture of independently shifted slots.', '',
                  '## Strict safety and distribution', '',
                  'Strict selection requires per-context FRC >= baseline and exact FRD <= baseline, with no quality slack. Baseline fallback is always eligible. It maximizes FRC only within the finite proposed family, not over all possible corrections.', '',
                  f"Strict selected sources: `{json.dumps(selections['strict_oracle'], sort_keys=True)}`.", '',
                  '| Method / metric | Mean delta | P10 / median / P90 | Session bootstrap95% CI |', '|---|---:|---|---|']
        for arm, metrics in differences.items():
            for key in ['FRC', 'exact_FRD', 'raw_FRC']:
                d = metrics[key]; q = d['quantiles_P0_P10_P50_P90_P100']; ci = d['session_bootstrap_CI95']
                lines.append(f"| {arm} / {key} | {d['mean']:+.6f} | {q[1]:+.6f} / {q[2]:+.6f} / {q[3]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
        lines += ['', 'Bootstrap describes variation across evaluated sessions, not generalization of an Oracle to unseen GT. Full per-session metrics and context ledgers are retained locally.', '',
                  '## Invariance, scope and cost', '',
                  f"Maximum centered-geometry change: {summary['geometry_max_error']:.3g}; each context's native FRDiv change is asserted below1e-6. AU/expression are bitwise unchanged. Common VA shifts stay in[-1,1] without independent clipping.", '',
                  'q0 VA output is also corrected; its model weights remain frozen. An unchanged-q0-output guarantee is not claimed. Raw-GT FRC is diagnostic and is not a constraint used to select these Oracle sets.', '',
                  'Baseline uses ten3,720,129-parameter teachers. The GT-free correction uses an additional ten-model T0_2400 ensemble:74,402,580 parameters total, final output10. No weights are trained. The oracle optimizes at most32 per-context control parameters for100 Adam steps; these are GT-dependent, not deployable model weights.', '',
                  f"Cached-prediction correction overhead: {1000*summary['speed']['gt_free_seconds']:.3f}ms/context; oracle proposal optimization: {summary['speed']['oracle_seconds']:.3f}s/context; exact scoring of five sets: {summary['speed']['exact_FRD_seconds']:.3f}s/context.", '',
                  'These timings exclude baseline/quality model forward passes and are not end-to-end inference latency. Exact DTW reuses invariant AU/expression costs and uses the same nonnegative-group pruning and rolling recurrence as Mam. Parity is tested and the baseline FRD is reproduced for every context.', '',
                  f"Base commit: `{provenance['base_git_sha']}` plus frozen source hashes. Summary implementation SHA256: `{summary['source_sha256']}`.", '',
                  'No checkpoint, TRAIN target or student cache is replaced. This is a local diagnostic, not a new official TEST result.', '',
                  'See [the prespecified protocol](COMMON_VA_PROTOCOL.md).']
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'contexts': len(rows), 'metrics': summaries, 'selection': selections}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['generate', 'score', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    for name in ['baseline', 'quality', 'target-cache', 'data-root', 'asset-mam-root', 'report']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--phase', choices=['pilot', 'full'], default='full')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--workers', type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(4)
    {'generate': generate, 'score': score, 'summarize': summarize}[args.action](args)


if __name__ == '__main__':
    main()
