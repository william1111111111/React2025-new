"""Aggregate the matched contrast/AU student continuation experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.analyze_t1_student_distillation import geometry, alignment, mean_rows


def paired_deltas(rows, reference):
    answer = {}
    for metric in ('FRC', 'exact_FRD', 'FRDiv'):
        values = lambda r: sum(r['frc']) if metric == 'FRC' else r[metric]
        differences = np.array([values(r)-values(b) for r, b in zip(rows, reference)])
        sessions = {}
        for row, delta in zip(rows, differences):
            sessions.setdefault(row['clip_id'].split('/')[1], []).append(delta)
        sums = np.array([sum(v) for v in sessions.values()])
        counts = np.array([len(v) for v in sessions.values()])
        draws = np.random.default_rng(1234).integers(0, len(sessions), size=(5000, len(sessions)))
        bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
        answer[metric] = {'mean_delta': float(differences.mean()),
                          'session_bootstrap_CI95': np.quantile(bootstrap, [.025, .975]).tolist(),
                          'context_improvement_fraction': float(np.mean(
                              differences < 0 if metric == 'exact_FRD' else differences > 0))}
    return answer


def au_diagnostics(predictions, teachers):
    bins = ((0., .05), (.05, .20), (.20, .501))
    totals = np.zeros((3, 2), dtype=np.int64)
    false_positive = false_negative = teacher_on = teacher_off = 0
    for prediction, teacher in zip(predictions, teachers):
        p, t = prediction[..., :15], teacher[..., :15]
        mismatch = p.round() != t.round()
        margin = (t-.5).abs()
        for index, (low, high) in enumerate(bins):
            selected = (margin >= low) & (margin < high)
            totals[index] += [int((selected & mismatch).sum()), int(selected.sum())]
        on = t.round().bool()
        false_negative += int((on & ~p.round().bool()).sum())
        false_positive += int((~on & p.round().bool()).sum())
        teacher_on += int(on.sum())
        teacher_off += int((~on).sum())
    return {'teacher_relative_false_negative_rate': false_negative/max(1, teacher_on),
            'teacher_relative_false_positive_rate': false_positive/max(1, teacher_off),
            'teacher_margin_bins': [{'min': lo, 'max': hi, 'disagreement': int(n), 'count': int(d),
                                    'disagreement_fraction': int(n)/max(1, int(d))}
                                   for (lo, hi), (n, d) in zip(bins, totals)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--parent-run-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, parent = args.run_root, args.parent_run_root
    matched = json.loads((root/'matched_summary.json').read_text())
    if matched['matched_checks'] != 'passed':
        raise ValueError('Matched checks required')
    previous = json.loads((parent/'analysis.json').read_text())
    parent_eval = json.loads((parent/'contrast/val571_full.json').read_text())
    teacher_prediction = torch.load(parent/'teacher_val571_fast.predictions.pt', map_location='cpu')
    reference_keys = [(r['clip_id'], r['target_cache_key']) for r in parent_eval['rows']]
    metrics = {'T1_teacher': previous['metrics']['teacher'],
               'contrast_epoch10': parent_eval['metrics']}
    results = {'contrast_epoch10': parent_eval}
    sources = {'contrast_epoch10': parent/'contrast/val571_full.predictions.pt'}
    for arm in matched['metrics']:
        result = json.loads((root/arm/'val571_full.json').read_text())
        if [(r['clip_id'], r['target_cache_key']) for r in result['rows']] != reference_keys:
            raise ValueError(f'Context/GT protocol mismatch for {arm}')
        if result['metrics']['contexts'] != 571:
            raise ValueError('This report requires full VAL571')
        metrics[arm], results[arm] = result['metrics'], result
        sources[arm] = root/arm/'val571_full.predictions.pt'
    diagnostics = {}
    for arm, path in sources.items():
        predictions = torch.load(path, map_location='cpu')
        if predictions['clip_ids'] != teacher_prediction['clip_ids']:
            raise ValueError('Teacher/student prediction order mismatch')
        official = mean_rows([geometry(x) for x in predictions['official_predictions']])
        if abs(official['FRDiv']-metrics[arm]['FRDiv']) > 1e-5:
            raise ValueError('Official diversity decomposition mismatch')
        frdiv = np.array([r['FRDiv'] for r in results[arm]['rows']])
        diagnostics[arm] = {
            'official_geometry': official,
            'alignment': mean_rows([alignment(x, y) for x, y in zip(
                predictions['soft_predictions'], teacher_prediction['soft_predictions'])]),
            'au_decisions': au_diagnostics(predictions['soft_predictions'], teacher_prediction['soft_predictions']),
            'FRDiv_quantiles_P0_P10_P50_P90_P100': np.quantile(frdiv, [0, .1, .5, .9, 1]).tolist(),
            'versus_epoch10': paired_deltas(results[arm]['rows'], parent_eval['rows']),
            'versus_matched_control': paired_deltas(results[arm]['rows'], results['control']['rows'])}
    teacher = metrics['T1_teacher']
    safe = [arm for arm in sources if metrics[arm]['FRC'] >= teacher['FRC']
            and metrics[arm]['exact_FRD'] <= teacher['exact_FRD']]
    safe_vs_parent = [arm for arm in sources if metrics[arm]['FRC'] >= metrics['contrast_epoch10']['FRC']
                      and metrics[arm]['exact_FRD'] <= metrics['contrast_epoch10']['exact_FRD']]
    selection = {'quality_preserving_vs_T1': safe,
                 'quality_preserving_vs_parent': safe_vs_parent,
                 'best_quality_preserving_vs_T1': max(safe, key=lambda a: metrics[a]['FRDiv']) if safe else None,
                 'best_quality_preserving_vs_parent': max(safe_vs_parent, key=lambda a: metrics[a]['FRDiv']) if safe_vs_parent else None,
                 'largest_FRDiv': max(sources, key=lambda a: metrics[a]['FRDiv'])}
    summary = {'metrics': metrics, 'selection': selection, 'diagnostics': diagnostics,
               'matched': matched,
               'checkpoint_hashes': {arm: r['checkpoint_sha256'] for arm, r in results.items()},
               'evaluator_hashes': {arm: r.get('evaluator_sha256') for arm, r in results.items()},
               'teacher_reference_analysis_sha256': hashlib.sha256((parent/'analysis.json').read_bytes()).hexdigest(),
               'protocol': 'REACT2025 full VAL571; all-session GT; center750; AU rounded; exact rolling FRD',
               'limitations': ['single seed', 'adaptive VAL experiment, not held-out TEST',
                               'T1 teacher is a 600-step pilot; not equivalent to T0 quality',
                               'frozen teacher metrics reused from parity-verified parent experiment',
                               'reported latency is diagnostic, not a controlled GPU benchmark']}
    (root/'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# T1 student: quality-preserving AU refinement', '', summary['protocol'], '',
             '| Method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | T1 FRDiv retained |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, m in metrics.items():
        lines.append(f"| {arm} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} | {100*m['FRDiv']/teacher['FRDiv']:.1f}% |")
    lines += ['', '## Experiment contract', '',
              '- Frozen starting point: the previous 10-epoch centered-residual student. Every new arm starts from exactly those weights.',
              '- Three arms: unchanged contrast-KD control; contrast KD + confidence-weighted AU decisions; contrast KD + signed AU log-odds contrasts.',
              '- All arms: five more epochs / 4,150 updates, seed 1, batch 4, lr 2e-5 → 1e-6 cosine, BF16, clip norm 1.',
              '- AdamW is reset identically in every arm: betas (0.9,0.99), weight decay 1e-4. This is matched fine-tuning, not an optimizer resume.',
              '- Frozen base loss: MSE + (1-CCC) + 0.05 velocity + 0.2697612303 centered-residual loss, with unchanged TRAIN scales.',
              '- Extra loss weights use 16 fixed TRAIN probes and target 0.25 of the initial primary-loss gradient norm. No VAL labels enter training/calibration.',
              '- AU logit contrast: clip probabilities to [0.001,0.999], subtract the ten-slot logit mean, match detached teacher contrasts with Huber loss.',
              '- Same 3.803M parameters, ten outputs, teacher cache, initial-state hash and observed source-order hash. No inference noise/scaling or teacher calls.',
              '- Each student now has 12,450 total training updates. Comparisons against epoch10 alone are NOT budget-matched; the continuation control is the causal comparator.', '',
              '## Selection and caution', '',
              f"Highest FRDiv: `{selection['largest_FRDiv']}`.",
              f"Highest FRDiv satisfying both frozen T1 quality point estimates: `{selection['best_quality_preserving_vs_T1']}`.",
              f"Highest FRDiv satisfying both frozen parent-student quality point estimates: `{selection['best_quality_preserving_vs_parent']}`.",
              f"Arms preserving both parent-student quality point estimates: {', '.join(safe_vs_parent) or 'none'}.",
              'No quality slack is used. Passing the teacher thresholds does not imply passing the stronger parent-student thresholds.',
              'This is a single-seed adaptive VAL study, not proof of statistical equivalence or a TEST result.', '',
              '## Deltas versus matched continuation control', '',
              '| Arm | ΔFRC [session bootstrap 95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |',
              '|---|---|---|---|']
    for arm in matched['metrics']:
        cells = []
        for metric in ('FRC', 'exact_FRD', 'FRDiv'):
            item = diagnostics[arm]['versus_matched_control'][metric]
            lo, hi = item['session_bootstrap_CI95']
            cells.append(f"{item['mean_delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        lines.append(f"| {arm} | " + ' | '.join(cells) + ' |')
    if all(metrics[a]['FRDiv'] <= metrics['control']['FRDiv'] for a in ('au_decision', 'au_logit')):
        lines += ['', 'Both AU auxiliary objectives fail to improve FRDiv over the equal-budget continuation control.',
                  'Any diversity gain versus epoch10 therefore cannot be attributed to these new objectives.']
    lines += ['', '## AU fidelity and diversity components', '',
              '| Arm | AU FRDiv | VA FRDiv | Expression FRDiv | AU decision disagreement | AU contrast cosine |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, diag in diagnostics.items():
        g, a = diag['official_geometry'], diag['alignment']
        lines.append(f"| {arm} | {g['AU_total']:.6f} | {g['VA_total']:.6f} | {g['expression_total']:.6f} | {a['AU_rounding_disagreement_fraction']:.2%} | {a['AU_contrast_cosine']:.6f} |")
    lines += ['', '### Teacher-relative AU errors', '',
              'These are disagreement rates with teacher predictions, NOT accuracy against real GT. Margin-bin rates pool all valid evaluated frames.', '',
              '| Arm | Teacher-on missed | Teacher-off falsely activated | Disagreement at teacher margin < 0.05 |',
              '|---|---:|---:|---:|']
    for arm, diag in diagnostics.items():
        au = diag['au_decisions']
        lines.append(f"| {arm} | {au['teacher_relative_false_negative_rate']:.2%} | {au['teacher_relative_false_positive_rate']:.2%} | {au['teacher_margin_bins'][0]['disagreement_fraction']:.2%} |")
    lines += ['', '## Compute and provenance', '',
              '| Arm | Extra weight | Updates | Train seconds | Updates/s | Diagnostic inference ms |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, config in matched['configs'].items():
        seconds = config['training_seconds']
        lines.append(f"| {arm} | {config['extra_weight']:.6f} | {config['updates']} | {seconds:.1f} | {config['updates']/seconds:.2f} | {1000*results[arm]['speed']['inference_latency_s_mean']:.2f} |")
    lines += ['', 'No expensive temporal-GT loss or inference selector was added. Selector overhead is zero.',
              'All original/new weights are retained. Per-context metrics, distributions, teacher-margin-bin errors and hashes are saved locally in analysis.json.',
              'Teacher metrics are reused from the preceding parity-verified evaluation. Its 600-step T1 ensemble already sacrifices quality versus T0; do not conflate their quality references.', '']
    for arm, prov in matched['provenance'].items():
        lines += [f"- {arm}: checkpoint `{summary['checkpoint_hashes'][arm]}`, initial state `{prov['initial_student_state_sha256']}`, source order `{prov['observed_source_schedule_sha256']}`."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics': metrics, 'selection': selection}, indent=2))


if __name__ == '__main__':
    main()
