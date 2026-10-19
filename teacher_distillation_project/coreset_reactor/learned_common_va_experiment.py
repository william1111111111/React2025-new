"""Generate and exactly score the frozen learned common-VA policies."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.common_va import apply_shift, SharedVAQualityPool
from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.learned_common_va import CommonVACorrector, feasible_batch_shift
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import sha256_file


POLICIES = ('baseline', 'ungated', 'gate050', 'gate080', 'gate090')
THRESHOLDS = {'gate050': .5, 'gate080': .8, 'gate090': .9}
ROOT = Path(__file__).resolve().parents[1]


def load_target(cache, row):
    key = row['target_cache_key']
    state = torch.load(cache/f'{key}.pt', map_location='cpu')
    if state['fingerprint'] != key:
        raise ValueError('Frozen target cache mismatch')
    return state['targets']


def generate(args):
    args.root.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    predictions = torch.load(args.baseline.with_suffix('.predictions.pt'), map_location='cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = CommonVACorrector(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model.to(device).eval()
    data = PairedReactionDataset(args.data_root, 'val', 750, crop_mode='center',
                                 normalization_dir=args.asset_mam_root/'external/FaceVerse')
    if len(data) != len(baseline['rows']) or predictions['clip_ids'] != [x['clip_id'] for x in baseline['rows']]:
        raise ValueError('Baseline/data identities do not align')
    proposals, latencies = [], []
    with torch.inference_mode():
        for index in range(len(data)):
            sample, prediction = data[index], predictions['official_predictions'][index]
            length = int(sample['source_lengths'])
            if sample['clip_id'] != predictions['clip_ids'][index] or prediction.shape != (10, length, 25):
                raise ValueError(f'Prediction identity/shape mismatch at {index}')
            mean = torch.zeros(1, 750, 25); std = torch.zeros_like(mean)
            mean[0, :length] = prediction.mean(0)
            std[0, :length] = prediction.std(0, unbiased=False)
            inputs = [sample[x][None].to(device) for x in ('speaker_audio','speaker_emotion','speaker_3dmm')]
            if device.type == 'cuda': torch.cuda.synchronize(device)
            tick = time.perf_counter()
            result = model(*inputs, mean.to(device), std.to(device), torch.tensor([length], device=device))
            desired = result['shift'][:, :length].float()
            shift = feasible_batch_shift(prediction[None].to(device), desired)[0].cpu()
            probability = float(result['gate_logit'].sigmoid())
            if device.type == 'cuda': torch.cuda.synchronize(device)
            latencies.append(time.perf_counter()-tick)
            proposals.append({'index': index, 'clip_id': sample['clip_id'], 'length': length,
                              'shift': shift.half(), 'gate_probability': probability})
            if (index+1) % 50 == 0 or index+1 == len(data):
                print(json.dumps({'completed': index+1, 'total': len(data)}), flush=True)
    torch.save({'proposals': proposals, 'policies': POLICIES, 'thresholds': THRESHOLDS},
               args.root/'proposals.pt')
    source_files = ['coreset_reactor/learned_common_va.py',
                    'coreset_reactor/learned_common_va_experiment.py',
                    'coreset_reactor/results/LEARNED_COMMON_VA_PROTOCOL.md']
    provenance = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': {x: sha256_file(ROOT/x) for x in source_files},
                  'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256_file(args.checkpoint),
                  'baseline': str(args.baseline.resolve()), 'baseline_sha256': sha256_file(args.baseline),
                  'baseline_predictions_sha256': sha256_file(args.baseline.with_suffix('.predictions.pt')),
                  'target_cache': str(args.target_cache.resolve()), 'data_root': str(args.data_root.resolve()),
                  'asset_mam_root': str(args.asset_mam_root.resolve()), 'contexts': len(data),
                  'parameter_count': checkpoint['parameter_count'], 'policies': POLICIES,
                  'train_seconds': checkpoint['train_seconds'],
                  'train_label_oracle_summary': checkpoint['train_label_oracle_summary'],
                  'thresholds': THRESHOLDS, 'primary_policy': 'gate080',
                  'inference_latency_seconds_mean': float(np.mean(latencies)),
                  'inference_latency_seconds_median': float(np.median(latencies)),
                  'listener_GT_available_to_model': False, 'validation_tuning': False}
    (args.root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def score(args):
    from framework.metrics import compute_FRVar, compute_s_mse
    provenance = json.loads((args.root/'provenance.json').read_text())
    baseline_path, cache = Path(provenance['baseline']), Path(provenance['target_cache'])
    baseline = json.loads(baseline_path.read_text())
    predictions = torch.load(baseline_path.with_suffix('.predictions.pt'), map_location='cpu')
    proposals = torch.load(args.root/'proposals.pt', map_location='cpu')['proposals']
    data = PairedReactionDataset(Path(provenance['data_root']), 'val', 750, crop_mode='center',
                                 normalization_dir=Path(provenance['asset_mam_root'])/'external/FaceVerse')
    facial = Path(provenance['data_root'])/'val/facial-attributes'
    pools = {}
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools.setdefault(path.parent.name, []).append(path)
    ledger = args.root/'rows'; ledger.mkdir(exist_ok=True)
    indices = list(range(len(baseline['rows'])))[args.shard_index::args.shard_count]
    pool = SharedVAQualityPool(args.workers, max(x['gt_count'] for x in baseline['rows']))
    started = time.monotonic()
    try:
        for completed, index in enumerate(indices, 1):
            output = ledger/f'{index:06d}.json'
            if output.exists(): raise FileExistsError(output)
            original, proposal = baseline['rows'][index], proposals[index]
            pred = predictions['official_predictions'][index]
            if not (original['clip_id'] == proposal['clip_id'] == predictions['clip_ids'][index]):
                raise ValueError('Identity mismatch')
            target = load_target(cache, original)
            shift = proposal['shift'].float()
            variants = [pred, apply_shift(pred, shift)]
            variants += [apply_shift(pred, shift) if proposal['gate_probability'] >= THRESHOLDS[name] else pred
                         for name in ('gate050','gate080','gate090')]
            frc = candidate_frc(target.numpy(), torch.cat(variants).numpy()).reshape(len(POLICIES), 10)
            frd, calls = pool.score_variants(variants, target)
            np.testing.assert_allclose(frc[0], original['frc'], atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(frd[0].sum(), original['exact_FRD'], atol=1e-7, rtol=1e-8)
            sample = data[index]
            length, start = int(sample['source_lengths']), int(sample['crop_start'])
            paired = (facial/sample['clip_id'].replace('speaker/','listener/',1)).with_suffix('.npy')
            gt_paths = [paired, *[x for x in pools[sample['session_id']] if x != paired]]
            raw = [torch.from_numpy(np.load(x, allow_pickle=False).astype(np.float32)) for x in gt_paths]
            raw_target = aligned_targets(raw, length, start)
            raw_frc = candidate_frc(raw_target.numpy(), torch.cat(variants).numpy()).reshape(len(POLICIES),10).sum(1)
            metrics = {}
            for arm, value, frc_arm, frd_arm, raw_arm in zip(POLICIES, variants, frc, frd, raw_frc):
                centered_error = float(((value.double()-value.double().mean(0))-
                                        (pred.double()-pred.double().mean(0))).abs().max())
                metrics[arm] = {'FRC': float(frc_arm.sum()), 'frc': frc_arm.tolist(),
                                'exact_FRD': float(frd_arm.sum()), 'frd_per_rank': frd_arm.tolist(),
                                'FRDiv': float(compute_s_mse([value])), 'FRVar': float(compute_FRVar([value])),
                                'raw_FRC': float(raw_arm), 'geometry_max_error': centered_error}
            if max(abs(x['FRDiv']-metrics['baseline']['FRDiv']) for x in metrics.values()) >= 1e-6:
                raise ValueError('Common correction changed FRDiv')
            row = {'index': index, 'clip_id': original['clip_id'], 'session_id': sample['session_id'],
                   'target_cache_key': original['target_cache_key'], 'gt_count': len(target),
                   'gate_probability': proposal['gate_probability'], 'metrics': metrics,
                   'dtw_group_calls': calls.tolist()}
            output.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
            if completed % 10 == 0 or completed == len(indices):
                progress = {'completed': completed, 'total': len(indices),
                            'elapsed_seconds': time.monotonic()-started}
                (args.root/f'shard{args.shard_index}.progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
    finally:
        pool.close()


def paired_summary(values, sessions):
    values = np.asarray(values, dtype=np.float64)
    grouped = {}
    for session, value in zip(sessions, values): grouped.setdefault(session, []).append(value)
    sums, counts = np.array([sum(x) for x in grouped.values()]), np.array([len(x) for x in grouped.values()])
    draws = np.random.default_rng(1234).integers(0, len(sums), (5000, len(sums)))
    boot = sums[draws].sum(1)/counts[draws].sum(1)
    return {'mean': float(values.mean()), 'P0_P10_P50_P90_P100': np.quantile(values,[0,.1,.5,.9,1]).tolist(),
            'positive_fraction': float(np.mean(values > 0)),
            'session_bootstrap_CI95': np.quantile(boot,[.025,.975]).tolist()}


def summarize(args):
    provenance = json.loads((args.root/'provenance.json').read_text())
    rows = [json.loads((args.root/f'rows/{i:06d}.json').read_text()) for i in range(provenance['contexts'])]
    keys = ('FRC','exact_FRD','FRDiv','FRVar','raw_FRC')
    metrics = {arm: {key: float(np.mean([r['metrics'][arm][key] for r in rows])) for key in keys}
               for arm in POLICIES}
    sessions = [r['session_id'] for r in rows]
    deltas = {arm: {key: paired_summary([r['metrics'][arm][key]-r['metrics']['baseline'][key] for r in rows], sessions)
                    for key in keys} for arm in POLICIES[1:]}
    pass_rates = {arm: float(np.mean([r['metrics'][arm]['FRC'] >= r['metrics']['baseline']['FRC'] and
                                      r['metrics'][arm]['exact_FRD'] <= r['metrics']['baseline']['exact_FRD']
                                      for r in rows])) for arm in POLICIES[1:]}
    per_session = {}
    for session in sorted(set(sessions)):
        selected = [r for r in rows if r['session_id'] == session]
        per_session[session] = {'contexts': len(selected), **{arm: {key: float(np.mean([r['metrics'][arm][key] for r in selected])) for key in keys} for arm in POLICIES}}
    oracle_diag = None
    if args.oracle_root:
        oracle_rows = [json.loads((args.oracle_root/f'rows/{i:06d}.json').read_text()) for i in range(len(rows))]
        truth = np.array([r['strict_choice'] != 'baseline' for r in oracle_rows])
        oracle_diag = {}
        for arm, threshold in THRESHOLDS.items():
            predicted = np.array([r['gate_probability'] >= threshold for r in rows])
            tp = int((predicted & truth).sum())
            oracle_diag[arm] = {'threshold': threshold, 'predicted_positive': int(predicted.sum()),
                                'oracle_positive': int(truth.sum()), 'precision': tp/max(1,int(predicted.sum())),
                                'recall': tp/max(1,int(truth.sum()))}
    summary = {'contexts': len(rows), 'metrics': metrics, 'paired_deltas': deltas,
               'per_context_FRC_and_FRD_nonworse_fraction': pass_rates,
               'gate_rates': {arm: float(np.mean([r['gate_probability'] >= t for r in rows])) for arm,t in THRESHOLDS.items()},
               'gate_probability_quantiles': np.quantile([r['gate_probability'] for r in rows],[0,.1,.5,.9,1]).tolist(),
               'oracle_gate_diagnostic': oracle_diag, 'per_session': per_session,
               'max_geometry_error': max(r['metrics'][arm]['geometry_max_error'] for r in rows for arm in POLICIES),
               'dtw_group_calls': np.sum([r['dtw_group_calls'] for r in rows],0).tolist(),
               'primary_policy': 'gate080', 'primary_strict_success':
                   metrics['gate080']['FRC'] >= metrics['baseline']['FRC'] and
                   metrics['gate080']['exact_FRD'] <= metrics['baseline']['exact_FRD'] and
                   abs(metrics['gate080']['FRDiv']-metrics['baseline']['FRDiv']) < 1e-6 and
                   metrics['gate080']['raw_FRC'] >= metrics['baseline']['raw_FRC'],
               'provenance': provenance}
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = ['# Learned inference-safe common-VA correction', '',
                 'REACT2025 full VAL571, all-session processed-GT extension. The corrector was fitted only on TRAIN1660 labels; no listener GT, T0 model, metric, oracle, or tuned threshold is available at inference.', '',
                 '| Policy | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ | Gate rate |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for arm in POLICIES:
            m=metrics[arm]; gate='0.000000' if arm=='baseline' else '1.000000' if arm=='ungated' else f"{summary['gate_rates'][arm]:.6f}"
            lines.append(f"| {arm}{' **(primary)**' if arm=='gate080' else ''} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} | {m['raw_FRC']:.6f} | {gate} |")
        primary=metrics['gate080']; base=metrics['baseline']
        lines += ['', '## Prespecified decision', '',
                  f"Primary strict success: **{summary['primary_strict_success']}**. Delta FRC {primary['FRC']-base['FRC']:+.6f}; exact FRD {primary['exact_FRD']-base['exact_FRD']:+.6f}; FRDiv {primary['FRDiv']-base['FRDiv']:+.9f}; raw-GT FRC {primary['raw_FRC']-base['raw_FRC']:+.6f}.", '',
                  'Threshold0.80 and all training hyperparameters were frozen before label generation. Other policies are reported diagnostics, not replacements chosen after observing VAL.', '',
                  '## Scope', '',
                  f"TRAIN finite-oracle labels: `{json.dumps(provenance['train_label_oracle_summary'], sort_keys=True)}`.", '',
                  f"Corrector parameters: {provenance['parameter_count']:,}; training time {provenance['train_seconds']/60:.2f}min; cached-input latency {1000*provenance['inference_latency_seconds_mean']:.3f}ms/context. Pairwise geometry maximum numerical change {summary['max_geometry_error']:.3g}; AU/expression are unchanged and FRDiv is structurally invariant.", '',
                  f"Checkpoint SHA256: `{provenance['checkpoint_sha256']}`. Commit at execution: `{provenance['git_sha']}`.", '',
                  'This one-seed VAL result is exploratory, not an independent TEST claim. Full context ledgers, per-session metrics and session-bootstrap intervals are retained in the local artifact.']
        args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics': metrics, 'primary_success': summary['primary_strict_success'],
                      'gate_rates': summary['gate_rates']}, indent=2), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('generate','score','summarize'))
    p.add_argument('--root', type=Path, required=True)
    for name in ('checkpoint','baseline','target-cache','data-root','asset-mam-root','oracle-root','report'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--workers',type=int,default=10)
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1)
    args=p.parse_args(); torch.set_num_threads(4)
    {'generate':generate,'score':score,'summarize':summarize}[args.action](args)


if __name__ == '__main__': main()
