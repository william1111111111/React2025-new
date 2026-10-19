"""Raw-GT diagnostic and complete matched T1H result, including failed guards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.analyze_t1_student_refinement import paired_deltas
from coreset_reactor.diagnose_teacher_target_fit import aligned_targets
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.teacher_cache import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    provenance = json.loads((args.root/'provenance.json').read_text())
    assert provenance['matched_training_checks'] == 'passed'
    paths = {k: Path(v) for k, v in provenance['paths'].items()}
    baseline_path = paths['convergence_root']/'eval/T1_s000600_val571_full.json'
    baseline = json.loads(baseline_path.read_text())
    result = json.loads((args.root/'val571_full.json').read_text())
    prior = json.loads((paths['previous_research_root']/'val_representation_diagnostic.json').read_text())
    ids = [(r['clip_id'], r['target_cache_key']) for r in baseline['rows']]
    assert len(ids) == 571
    assert [(r['clip_id'], r['target_cache_key']) for r in result['rows']] == ids
    assert [r['clip_id'] for r in prior['rows']] == [r[0] for r in ids]
    for rank in range(1, 10):
        assert sha256_file(args.root/f'rank_{rank:02d}/conditional-step000600-seed1.pth') == result['checkpoint_hashes'][rank]
    assert result['checkpoint_hashes'][0] == provenance['q0_checkpoint_sha256']
    prediction = torch.load(args.root/'val571_full.predictions.pt', map_location='cpu')
    dataset = PairedReactionDataset(paths['data_root'], 'val', 750, crop_mode='center',
                                    normalization_dir=paths['asset_mam_root']/'external/FaceVerse')
    facial = paths['data_root']/'val/facial-attributes'
    pools = {}
    for path in sorted((facial/'listener').glob('*/*.npy')):
        pools.setdefault(path.parent.name, []).append(path)
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample['clip_id'] == ids[index][0] == prediction['clip_ids'][index]
        length, start = int(sample['source_lengths']), int(sample['crop_start'])
        paired = (facial/sample['clip_id'].replace('speaker/', 'listener/', 1)).with_suffix('.npy')
        gt_paths = [paired, *[x for x in pools[sample['session_id']] if x != paired]]
        raw = [torch.from_numpy(np.load(x, allow_pickle=False).astype(np.float32)) for x in gt_paths]
        target = aligned_targets(raw, length, start).numpy()
        pred = prediction['official_predictions'][index].numpy()
        rows.append({'clip_id': sample['clip_id'],
                     'raw_all_session_FRC': float(candidate_frc(target, pred).sum()),
                     'raw_native_paired_FRC': float(candidate_frc(target[:1], pred).sum())})
        if len(rows) % 100 == 0:
            print(f'Raw VAL diagnostic {len(rows)}/571', flush=True)
    raw_frc = float(np.mean([r['raw_all_session_FRC'] for r in rows]))
    raw_baseline = prior['summary']['T1']['all_session_training_representation_FRC']
    a, b = result['metrics'], baseline['metrics']
    guards = {'FRC': a['FRC'] >= b['FRC'], 'exact_FRD': a['exact_FRD'] <= b['exact_FRD'],
              'FRDiv': a['FRDiv'] >= b['FRDiv'], 'raw_FRC': raw_frc >= raw_baseline}
    deltas = paired_deltas(result['rows'], baseline['rows'])
    summary = {'metrics': a, 'baseline_metrics': b, 'raw_FRC': raw_frc,
               'baseline_raw_FRC': raw_baseline, 'raw_FRC_delta': raw_frc-raw_baseline,
               'raw_native_paired_FRC': float(np.mean([r['raw_native_paired_FRC'] for r in rows])),
               'guards': guards, 'strict_guard_pass': all(guards.values()), 'paired_deltas': deltas,
               'matched_training_checks': 'passed', 'matched_evaluation_checks': 'passed',
               'source_sha256': sha256_file(Path(__file__)), 'baseline_eval_sha256': sha256_file(baseline_path),
               'evaluation_sha256': sha256_file(args.root/'val571_full.json'),
               'limitations': ['single seed, repeatedly observed VAL; no independent TEST',
                               'raw diagnostic is not the official-processed metric',
                               'TRAIN target cache uses GT/length seeds, not evaluator context seeds']}
    (args.root/'raw_val_diagnostic.json').write_text(json.dumps({'rows': rows, 'mean_FRC': raw_frc}, indent=2)+'\n')
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    prior_processed = json.loads((paths['previous_research_root']/'processed_pilot600/val571_full.json').read_text())['metrics']
    lines = ['# T1H: preserve raw GT plus a weak processed-target auxiliary', '',
             'REACT2025, seed1, 600 updates per teacher, full VAL571. All-session opposite-role GT with official AU rounding and learned target processing; not official random-10 GT sampling.', '',
             '| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | Raw-GT diagnostic FRC ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    for name, metrics, raw in [('T1 raw baseline', b, raw_baseline),
                                ('T1P processed replacement', prior_processed, prior['summary']['T1P']['all_session_training_representation_FRC']),
                                ('T1H raw + 0.10 processed', a, raw_frc)]:
        lines.append(f"| {name} | {metrics['FRC']:.6f} | {metrics['exact_FRD']:.6f} | {metrics['FRDiv']:.6f} | {metrics['FRVar']:.6f} | {raw:.6f} |")
    lines += ['', f"Strict four-metric guard: **{'PASS' if all(guards.values()) else 'FAIL'}**.", '',
              'Individual guards: '+', '.join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in guards.items())+'.', '',
              '| Difference vs T1 | Mean | Session-bootstrap 95% CI | Context improvement fraction |',
              '|---|---:|---|---:|']
    for key, d in deltas.items():
        low, high = d['session_bootstrap_CI95']
        lines.append(f"| {key} | {d['mean_delta']:+.6f} | [{low:+.6f}, {high:+.6f}] | {d['context_improvement_fraction']:.4f} |")
    lines += ['', f'Raw-GT diagnostic FRC change: {raw_frc-raw_baseline:+.6f}. This diagnostic is supplementary, not an alternative scoring protocol chosen to favor the result.', '',
              '## Interpretation', '',
              ('All prespecified guards pass at the mean level in this one-seed pilot. This is not an independent confirmation or evidence that the gains generalize across seeds.' if all(guards.values()) else
               'Do not promote this pilot: at least one prespecified guard fails. Report the quality/diversity tradeoff rather than claiming a universal improvement. No weight sweep or additional training followed inspection.'), '',
              '## Controls and compute', '',
              'The same T1 TRAIN GT identities, native paired anchor, architecture, initialization, actual source/crop schedule, batch32, BF16, AdamW1e-4 and original 50-epoch cosine prefix were retained. Initial-state/source hashes passed for all nine new specialists. No student training or old-checkpoint replacement.', '',
              '`L = L_raw + 0.10 * (MSE_processed + 1 - CCC_processed)`. Raw supervision is not diluted; the original cross-listener velocity mask is retained; no processed velocity term. Targets are packed internally only for loss transport; predictions remain 25-D.', '',
              'Nine new models ×600 updates =5400 new updates; with the reused q0 the matched ensemble budget is6000 updates. One backbone forward per source, with one additional inexpensive loss evaluation on the same GT identity. No extra SoftDTW, parameters, or inference-time processor for the model. The evaluator target processor is unchanged.', '']
    audits = [json.loads((args.root/f'rank_{r:02d}/conditional-step000600-seed1.json').read_text()) for r in range(1, 10)]
    lines += [f"Parameters: {result['speed']['parameters']:,} across ten teachers. Candidate/output size10; selector overhead0.",
              f"Mean training updates/s/model: {np.mean([600/x['elapsed_training_seconds'] for x in audits]):.3f}. Training wall time: {provenance['training_wall_seconds']:.1f}s.",
              f"Inference latency: {1000*result['speed']['inference_latency_s_mean']:.2f}ms/context; GT cache hits: {result['speed']['target_cache_hits']}/571. These concurrent-device timings are diagnostic, not isolated speed benchmarks.", '',
              'The previous 5240-entry TRAIN GT/length cache is reused. Its original preprocessing cost is not included in optimizer throughput; cache seeds differ from evaluation-context seeds. No VAL GT enters training.', '',
              f"Base commit: `{provenance['base_git_sha']}` plus recorded uncommitted-source hashes. Analysis SHA256: `{summary['source_sha256']}`.", '',
              'Full per-context diagnostics and checkpoints remain local. This is a single-seed VAL-selected study, not a held-out TEST claim.', '',
              'See [the protocol](HYBRID_TEACHER_PROTOCOL.md).']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
