"""Report all six matched teacher-convergence points, without cherry-picking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.analyze_t1_student_distillation import geometry, mean_rows
from coreset_reactor.analyze_t1_student_refinement import paired_deltas

STEPS = (600, 1200, 2400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.run_root
    summary = json.loads((root/'summary.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    if summary['matched_training_checks'] != 'passed' or summary['matched_evaluation_checks'] != 'passed':
        raise ValueError('Completed matched audits required')
    results = {f'{arm}_{step}': json.loads((root/'eval'/f'{arm}_s{step:06d}_val571_full.json').read_text())
               for step in STEPS for arm in ('T0', 'T1')}
    metrics = {k: v['metrics'] for k, v in results.items()}
    reference_keys = [(r['clip_id'], r['target_cache_key']) for r in results['T0_600']['rows']]
    diagnostics, parity, training = {}, {}, {}
    for step in STEPS:
        for arm in ('T0', 'T1'):
            label = f'{arm}_{step}'
            result = results[label]
            if result['metrics']['contexts'] != 571:
                raise ValueError('Full VAL571 required')
            if [(r['clip_id'], r['target_cache_key']) for r in result['rows']] != reference_keys:
                raise ValueError('Evaluation protocol changed')
            prediction = torch.load(root/'eval'/f'{arm}_s{step:06d}_val571_full.predictions.pt', map_location='cpu')
            group_geometry = mean_rows([geometry(x) for x in prediction['official_predictions']])
            if abs(group_geometry['FRDiv']-metrics[label]['FRDiv']) > 1e-5:
                raise ValueError('Diversity decomposition disagrees with official FRDiv')
            del prediction
            values = np.array([r['FRDiv'] for r in result['rows']])
            diagnostics[label] = {
                'official_geometry': group_geometry,
                'versus_same_arm_600': paired_deltas(result['rows'], results[f'{arm}_600']['rows']),
                'versus_T0_same_budget': paired_deltas(result['rows'], results[f'T0_{step}']['rows']),
                'FRDiv_quantiles_P0_P10_P50_P90_P100': np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
                'context_fraction_FRDiv_ge_0_20': float(np.mean(values >= .20)),
            }
            audits = [json.loads((root/'arms'/arm/f'rank_{rank:02d}'/
                                 f'conditional-step{step:06d}-seed1.json').read_text()) for rank in range(10)]
            training[label] = {
                'sum_training_seconds_to_step': sum(a['elapsed_training_seconds'] for a in audits),
                'mean_updates_per_second_per_model': float(np.mean([step/a['elapsed_training_seconds'] for a in audits])),
                'parameter_count_per_model': audits[0]['parameter_count'],
                'ensemble_parameter_count': sum(a['parameter_count'] for a in audits),
                'updates_per_model': step,
                'exact_temporal_slot_GT_budget': 'one fixed GT per source per independent teacher; unchanged',
            }
    for arm in ('T0', 'T1'):
        old = json.loads((args.frozen_root/f'eval_{arm}_val571.json').read_text())['metrics']
        parity[arm] = {
            'old_metrics': old,
            'new600_minus_old': {k: metrics[f'{arm}_600'][k]-old[k] for k in ('FRC', 'exact_FRD', 'FRDiv', 'FRVar')},
            'rank_parameter_parity': [json.loads((root/'arms'/arm/f'rank_{rank:02d}'/
                                                 'frozen600_parameter_parity.json').read_text()) for rank in range(10)]}
    baseline = metrics['T1_600']
    strict = [f'T1_{step}' for step in STEPS[1:] if
              metrics[f'T1_{step}']['FRC'] >= baseline['FRC'] and
              metrics[f'T1_{step}']['exact_FRD'] <= baseline['exact_FRD'] and
              metrics[f'T1_{step}']['FRDiv'] >= baseline['FRDiv']]
    strict_t0 = [f'T0_{step}' for step in STEPS[1:] if
                 metrics[f'T0_{step}']['FRC'] >= metrics['T0_600']['FRC'] and
                 metrics[f'T0_{step}']['exact_FRD'] <= metrics['T0_600']['exact_FRD'] and
                 metrics[f'T0_{step}']['FRDiv'] >= metrics['T0_600']['FRDiv']]
    quality_safe = [f'T1_{step}' for step in STEPS if
                    metrics[f'T1_{step}']['FRC'] >= baseline['FRC'] and
                    metrics[f'T1_{step}']['exact_FRD'] <= baseline['exact_FRD']]
    selection = {
        'reference': 'fresh matched T1_600; frozen historical600 reproduced separately',
        'strict_T1_Pareto_candidates': strict,
        'strict_T0_Pareto_candidates_versus_T0_600': strict_t0,
        'highest_diversity_quality_safe_T1': max(quality_safe, key=lambda k: metrics[k]['FRDiv']),
        'strict_definition': 'FRC >= baseline, exact_FRD <= baseline, FRDiv >= baseline; zero slack',
        'fresh_vs_historical_warning': 'BF16/CUDA reruns need not be bitwise identical; only fresh matched curves establish the training-length contrast',
    }
    diversity_changes = {}
    for before, after in ((600, 1200), (1200, 2400)):
        first = diagnostics[f'T1_{before}']['official_geometry']
        last = diagnostics[f'T1_{after}']['official_geometry']
        total = last['FRDiv']-first['FRDiv']
        dynamic = last['dynamic_FRDiv']-first['dynamic_FRDiv']
        diversity_changes[f'{before}_to_{after}'] = {
            'FRDiv_delta': total, 'dynamic_delta': dynamic,
            'static_delta': total-dynamic,
            'dynamic_share_of_delta': dynamic/total if abs(total) > 1e-12 else None,
        }
    report = {'protocol': 'REACT2025 full VAL571, center750, deterministic all-session GT, AU rounded, exact rolling FRD',
              'metrics': metrics, 'diagnostics': diagnostics, 'training': training,
              'T1_diversity_change_attribution': diversity_changes,
              'frozen600_reproduction': parity, 'selection': selection,
              'provenance': provenance,
              'analysis_source_sha256': {
                  path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__),
                               Path(__file__).with_name('analyze_t1_student_distillation.py'),
                               Path(__file__).with_name('analyze_t1_student_refinement.py'))},
              'evaluation_json_sha256': {
                  label: hashlib.sha256((root/'eval'/f'{label.split("_")[0]}_s{int(label.split("_")[1]):06d}_val571_full.json').read_bytes()).hexdigest()
                  for label in results},
              'limitations': ['one seed', 'repeated VAL inspection, not an independent TEST result',
                              'no new student distillation performed in this teacher-convergence experiment',
                              'latency is diagnostic with concurrent GPU work',
                              'strict inequalities refer to observed point estimates, not certified population-level dominance']}
    (root/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Matched T0/T1 teacher convergence: 600 → 1200 → 2400 updates', '',
             report['protocol'], '',
             '| Teacher | Updates/model | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar |',
             '|---|---:|---:|---:|---:|---:|']
    for step in STEPS:
        for arm in ('T0', 'T1'):
            m = metrics[f'{arm}_{step}']
            lines.append(f"| {arm} | {step} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
    lines += ['', '## Main decision', '',
              f"Strict zero-slack T1 improvements over the matched 600-update baseline: {', '.join(strict) or 'none'}.",
              f"Highest-FRDiv T1 checkpoint preserving baseline FRC and exact FRD: `{selection['highest_diversity_quality_safe_T1']}`.",
              f"T0 point-estimate improvements over its own matched 600-update baseline: {', '.join(strict_t0) or 'none'}.",
              'Do not equate outperforming T1_600 with reaching T0 quality. The matched T0 curve is shown at every budget.',
              'Any selected checkpoint is VAL-selected and single-seed; this is not independent confirmation.', '',
              '## Controlled protocol', '',
              '- Ten independent ConditionalREGNN teachers per arm; 3,720,129 parameters each, 37,201,290 per ensemble.',
              '- One fixed TRAIN target per source per teacher; original T0/T1 manifests are SHA256-verified. Rank0 remains the true paired anchor.',
              '- Seed1, batch32, BF16, AdamW lr1e-4, betas(0.9,0.99), wd1e-4, clip norm1, original 50-epoch cosine schedule.',
              '- MSE + (1-CCC) + 0.05 velocity, retaining original relative_time_masked policy: non-synchronous cross-listener velocity supervision remains disabled.',
              '- Fresh uninterrupted runs from the original initialization; no weight-only warmstart, scheduler reset or epoch-only resume.',
              '- Step snapshots do not consume RNG, update the optimizer or advance the scheduler. Initial states and observed source/crop-order hashes match across all20 teachers at every milestone.',
              '- Each model sees exactly600/1200/2400 updates at the reported checkpoints. The nominal50-epoch schedule is not a claim of50 completed epochs.',
              '- Twenty runs ×2400 updates =48000 new optimizer updates total. Relative to the600-step teacher point, the final point uses4× training updates; no increased-budget gain is called an architectural gain.',
              '- All six ensemble points receive full571-context exact FRD. No best-rank checkpoint mixing, inference scaling, noise or selector.',
              '- Original teachers and frozen student recipes remain untouched. This experiment does not retrain students.']
    for arm in ('T0', 'T1'):
        lines += ['', f'## {arm} learning-curve deltas versus fresh {arm}_600', '',
                  '| Updates | ΔFRC [session bootstrap95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |',
                  '|---:|---|---|---|']
        for step in STEPS[1:]:
            cells = []
            for key in ('FRC', 'exact_FRD', 'FRDiv'):
                d = diagnostics[f'{arm}_{step}']['versus_same_arm_600'][key]
                lo, hi = d['session_bootstrap_CI95']
                cells.append(f"{d['mean_delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            lines.append(f'| {step} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Per-rank FRC contribution', '',
              '| Arm / updates | q0 … q9 |', '|---|---|']
    for label, m in metrics.items():
        lines.append(f"| {label} | " + ', '.join(f'{v:.5f}' for v in m['FRC_per_expert']) + ' |')
    lines += ['', '## Output diversity composition', '',
              '| Arm / updates | AU | VA | Expression | Dynamic component | FRDiv P10 / P50 / P90 |',
              '|---|---:|---:|---:|---:|---|']
    for label, d in diagnostics.items():
        g, q = d['official_geometry'], d['FRDiv_quantiles_P0_P10_P50_P90_P100']
        lines.append(f"| {label} | {g['AU_total']:.6f} | {g['VA_total']:.6f} | {g['expression_total']:.6f} | {g['dynamic_FRDiv']:.6f} | {q[1]:.6f} / {q[2]:.6f} / {q[3]:.6f} |")
    lines += ['', '| T1 interval | Total diversity change | Dynamic change | Static change |',
              '|---|---:|---:|---:|']
    for interval, change in diversity_changes.items():
        lines.append(f"| {interval} | {change['FRDiv_delta']:+.6f} | {change['dynamic_delta']:+.6f} | {change['static_delta']:+.6f} |")
    lines += ['', 'Static means the temporal mean of each slot\'s deviation from the ensemble mean; dynamic is its time-varying residual. This algebraic decomposition does not certify semantic correctness or source compatibility.']
    lines += ['', '## Frozen600 reproduction', '',
              '| Arm | ΔFRC | Δexact FRD | ΔFRDiv | Bitwise-identical rank checkpoints |',
              '|---|---:|---:|---:|---:|']
    for arm, item in parity.items():
        d = item['new600_minus_old']
        count = sum(x['bitwise_identical'] for x in item['rank_parameter_parity'])
        lines.append(f"| {arm} | {d['FRC']:+.8f} | {d['exact_FRD']:+.8f} | {d['FRDiv']:+.8f} | {count}/10 |")
    lines += ['', 'Fresh600 is the causal baseline for the learning curve; parameter-level differences and all raw hashes are retained in the local ledger.', '',
              '## Compute / provenance', '',
              '| Arm / updates | Mean updates/s/model | Sum training seconds | Diagnostic ensemble inference ms |',
              '|---|---:|---:|---:|']
    for label, t in training.items():
        lines.append(f"| {label} | {t['mean_updates_per_second_per_model']:.2f} | {t['sum_training_seconds_to_step']:.1f} | {1000*results[label]['speed']['inference_latency_s_mean']:.2f} |")
    lines += ['', 'Candidate pool/output size:10. Temporal target budget:one fixed target per source/model. Selector overhead:zero.',
              f"Base commit: `{provenance['base_git_sha']}`; working-source hashes, target manifest hashes, initialization hashes and actual source/crop-order hashes are stored locally.",
              'Weights, target identities, trajectories and per-context evaluation data remain local; this report is aggregate-only.']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics': metrics, 'selection': selection}, indent=2))


if __name__ == '__main__':
    main()
