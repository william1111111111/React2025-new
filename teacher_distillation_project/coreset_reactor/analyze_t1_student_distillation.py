"""Summarize matched T1 KD runs and decompose retained output diversity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

GROUPS = {'AU': (0, 15), 'VA': (15, 17), 'expression': (17, 25)}


def geometry(prediction):
    """O(K) decomposition of official pairwise squared distances."""
    x = prediction.double()
    residual = x - x.mean(0, keepdim=True)
    static = residual.mean(1, keepdim=True)
    dynamic = residual - static
    factor = 2 * len(x) / (len(x) - 1)
    result = {}
    for name, (a, b) in GROUPS.items():
        # Contributions use the full 25-channel denominator and sum to FRDiv.
        for part, values in [('total', residual), ('static', static), ('dynamic', dynamic)]:
            result[f'{name}_{part}'] = float(factor * values[..., a:b].square().mean() * (b-a)/25)
    result['FRDiv'] = sum(result[f'{name}_total'] for name in GROUPS)
    result['dynamic_FRDiv'] = sum(result[f'{name}_dynamic'] for name in GROUPS)
    return result


def alignment(student, teacher):
    student, teacher = student.double(), teacher.double()
    sr = student - student.mean(0, keepdim=True)
    tr = teacher - teacher.mean(0, keepdim=True)
    result = {'trajectory_MSE_to_teacher': float((student - teacher).square().mean()),
              'mean_trajectory_MSE_to_teacher': float((student.mean(0) - teacher.mean(0)).square().mean()),
              'AU_rounding_disagreement_fraction': float((student[..., :15].round() != teacher[..., :15].round()).double().mean())}
    for name, (a, b) in GROUPS.items():
        x, y = sr[..., a:b], tr[..., a:b]
        result[f'{name}_contrast_cosine'] = float((x*y).sum() / (x.square().sum()*y.square().sum()).sqrt().clamp_min(1e-12))
        result[f'{name}_contrast_energy_ratio'] = float(x.square().sum()/y.square().sum().clamp_min(1e-12))
    return result


def mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--teacher-reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.run_root
    matched = json.loads((root/'matched_summary.json').read_text())
    teacher_eval = json.loads((root/'teacher_val571_fast.json').read_text())
    teacher_reference = json.loads(args.teacher_reference.read_text())
    teacher_predictions = torch.load(root/'teacher_val571_fast.predictions.pt', map_location='cpu')
    teacher = dict(teacher_eval['metrics'])
    reproduction_delta = {k: teacher[k] - teacher_reference['metrics'][k]
                          for k in ('FRC', 'FRDiv', 'FRVar')}
    if any(abs(v) > 1e-4 for v in reproduction_delta.values()):
        raise ValueError(f'Teacher reproduction failed: {reproduction_delta}')
    teacher['exact_FRD'] = teacher_reference['metrics']['exact_FRD']
    metrics, diagnostics, speeds, configs = {'teacher': teacher}, {}, {}, {}
    checkpoint_hashes = {'teacher': teacher_eval['checkpoint_sha256']}
    teacher_soft = teacher_predictions['soft_predictions']
    teacher_geometry = mean_rows([geometry(x) for x in teacher_predictions['official_predictions']])
    diagnostics['teacher'] = {'official_geometry': teacher_geometry,
                              'soft_geometry': mean_rows([geometry(x) for x in teacher_soft])}
    context_values, convergence = {}, {}
    arms = tuple(matched['metrics'])
    for arm in arms:
        result = json.loads((root/arm/'val571_full.json').read_text())
        predictions = torch.load(root/arm/'val571_full.predictions.pt', map_location='cpu')
        if predictions['clip_ids'] != teacher_predictions['clip_ids']:
            raise ValueError(f'Teacher/student context order differs: {arm}')
        if [r['target_cache_key'] for r in result['rows']] != [r['target_cache_key'] for r in teacher_eval['rows']]:
            raise ValueError(f'Teacher/student GT cache keys differ: {arm}')
        metrics[arm], speeds[arm] = result['metrics'], result['speed']
        checkpoint_hashes[arm] = result['checkpoint_sha256']
        configs[arm] = json.loads((root/arm/'config.json').read_text())
        mid = root/arm/'val571_epoch5_fast.json'
        convergence[arm] = {'epoch10': result['metrics']}
        if mid.is_file():
            convergence[arm]['epoch5'] = json.loads(mid.read_text())['metrics']
        official_geometry = [geometry(x) for x in predictions['official_predictions']]
        computed_frdiv = np.mean([x['FRDiv'] for x in official_geometry])
        if abs(computed_frdiv - metrics[arm]['FRDiv']) > 1e-5:
            raise ValueError('Diversity decomposition disagrees with official metric')
        frdiv = np.array([r['FRDiv'] for r in result['rows']])
        frc = np.array([sum(r['frc']) for r in result['rows']])
        frd = np.array([r['exact_FRD'] for r in result['rows']])
        context_values[arm] = {'FRC': frc, 'FRDiv': frdiv, 'exact_FRD': frd}
        diagnostics[arm] = {
            'official_geometry': mean_rows(official_geometry),
            'soft_geometry': mean_rows([geometry(x) for x in predictions['soft_predictions']]),
            'alignment': mean_rows([alignment(x, y) for x, y in zip(predictions['soft_predictions'], teacher_soft)]),
            'FRDiv_retention_vs_teacher': metrics[arm]['FRDiv']/teacher['FRDiv'],
            'FRDiv_quantiles_P0_P10_P50_P90_P100': np.quantile(frdiv, [0, .1, .5, .9, 1]).tolist(),
            'fraction_context_FRDiv_at_least_0_20': float(np.mean(frdiv >= .20)),
        }
    for arm in (a for a in arms if a != 'basic'):
        paired = {}
        for metric in ('FRC', 'FRDiv', 'exact_FRD'):
            delta = context_values[arm][metric] - context_values['basic'][metric]
            sessions = {}
            for clip, d in zip(teacher_predictions['clip_ids'], delta):
                sessions.setdefault(clip.split('/')[1], []).append(float(d))
            # Resample entire sessions; preserve context weighting inside each replicate.
            sums = np.array([sum(v) for v in sessions.values()])
            sizes = np.array([len(v) for v in sessions.values()])
            indices = np.random.default_rng(1234).integers(0, len(sessions), size=(5000, len(sessions)))
            boot = sums[indices].sum(1)/sizes[indices].sum(1)
            paired[metric] = {'mean_delta': float(delta.mean()),
                              'session_bootstrap_CI95': np.quantile(boot, [.025, .975]).tolist(),
                              'context_improvement_fraction': float(np.mean(delta < 0 if metric == 'exact_FRD' else delta > 0))}
        diagnostics[arm]['paired_vs_basic'] = paired
    quality_preserving = [name for name in arms
                          if metrics[name]['FRC'] >= teacher['FRC']
                          and metrics[name]['exact_FRD'] <= teacher['exact_FRD']]
    selection = {
        'highest_student_FRDiv': max(arms, key=lambda name: metrics[name]['FRDiv']),
        'quality_reference': 'frozen T1 teacher; FRC >= teacher and exact_FRD <= teacher',
        'quality_preserving_arms': quality_preserving,
        'recommended_quality_arm': (max(quality_preserving, key=lambda name: metrics[name]['FRDiv'])
                                    if quality_preserving else None),
    }
    summary = {'protocol': 'REACT2025 full VAL571, all-session GT, center750, AU-rounded; local analysis',
               'metrics': metrics, 'diagnostics': diagnostics, 'speed': speeds,
               'selection': selection,
               'convergence': convergence,
               'teacher_reproduction_delta': reproduction_delta,
               'teacher_reference_sha256': hashlib.sha256(args.teacher_reference.read_bytes()).hexdigest(),
               'checkpoint_hashes': checkpoint_hashes,
               'matched_provenance': matched['provenance'],
               'limitations': ['one seed', 'objectives explored on VAL, including an adaptive AU-decision follow-up',
                               'teacher is a frozen 600-step pilot',
                               'teacher exact FRD reused after FRC/FRDiv/FRVar reproduction',
                               'latency measurements are diagnostic and subject to concurrent GPU work']}
    (root/'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# T1 Pure Max-FRDiv teacher → shared student', '',
             summary['protocol'], '',
             '| Method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | Teacher FRDiv retained |',
             '|---|---:|---:|---:|---:|---:|']
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} | {100*m['FRDiv']/teacher['FRDiv']:.1f}% |")
    lines += ['', 'Teacher FRC/FRDiv/FRVar were reproduced using the student evaluator (maximum FRC difference 3.5e-7).',
              'Teacher exact FRD reuses its frozen full-VAL571 result; every student exact FRD was recomputed against all session GTs.']
    largest = selection['highest_student_FRDiv']
    lines += ['', '## Interpretation', '',
              f"Largest student FRDiv: `{largest}` = {metrics[largest]['FRDiv']:.6f} ({100*metrics[largest]['FRDiv']/teacher['FRDiv']:.1f}% retained)."]
    safe = selection['recommended_quality_arm']
    if safe:
        lines += [f"The highest-diversity student satisfying both T1 quality point-estimate thresholds is `{safe}`: "
                  f"FRC {metrics[safe]['FRC']:.6f}, exact FRD {metrics[safe]['exact_FRD']:.6f}, FRDiv {metrics[safe]['FRDiv']:.6f}."]
    else:
        lines += ['No student meets both T1 quality point-estimate thresholds.']
    lines += ['These are VAL comparisons from one seed; closeness in FRDiv alone does not establish lossless distillation.',
              'The old Prototype/KMeans teacher and its 0.0607 Basic-KD student are different target-generation experiments and are not the matched baseline here.']
    lines += ['', '## Matched training', '',
              '- One 3,803,201-parameter shared student, ten expert IDs; teacher ensemble: 37,201,290 parameters.',
              '- T1 rank-0 checkpoint initialization; seed 1, 3,320 fixed TRAIN views, batch 4, 10 epochs / 8,300 updates.',
              '- AdamW lr=1e-4, betas=(0.9,0.99), weight decay=1e-4; cosine schedule, BF16, clip norm 1.',
              '- Base KD: MSE + CCC + 0.05 velocity against soft teacher trajectories.',
              '- Distance KD matches 45 teacher pair distances within AU/VA/expression groups.',
              '- Contrast KD matches signed slot residuals about the set mean. It is exactly equivalent to matching pairwise difference vectors up to 2K/(K−1).',
              '- AU-decision KD is a follow-up motivated by the first three VAL results: confidence-weighted BCE against rounded TRAIN teacher AUs. Its weight is also calibrated on TRAIN only; it uses no inference scaling.',
              '- Auxiliary weights calibrated to 0.25 of initial KD gradient norm, using 32 TRAIN scale examples and 16 TRAIN gradient probes.',
              '- Student state hash, observed source order, teacher cache and evaluation GT keys matched across arms.',
              '- No inference spread scaling or added noise. All ten outputs come from the shared student.', '',
              '| Arm | Auxiliary weight | Training seconds | Updates/s | Inference ms (diagnostic) |',
              '|---|---:|---:|---:|---:|']
    for name, c in configs.items():
        seconds = c['training_seconds']
        lines.append(f"| {name} | {c['relation_weight']:.6f} | {seconds:.1f} | {8300/seconds:.2f} | {1000*speeds[name]['inference_latency_s_mean']:.2f} |")
    lines += ['', '## Convergence on full VAL571', '',
              '| Arm | Epoch | FRC | FRDiv | Soft FRDiv |',
              '|---|---:|---:|---:|---:|']
    for name, checkpoints in convergence.items():
        for epoch in ('epoch5', 'epoch10'):
            if epoch in checkpoints:
                m = checkpoints[epoch]
                lines.append(f"| {name} | {epoch[5:]} | {m['FRC']:.6f} | {m['FRDiv']:.6f} | {m['soft_FRDiv']:.6f} |")
    lines += ['', '## Diversity components', '',
              'AU/VA/expression contributions sum to official FRDiv. Dynamic contrast removes each slot’s time-average residual.', '',
              '| Method | AU | VA | Expression | Dynamic contrast | Soft FRDiv before AU rounding |',
              '|---|---:|---:|---:|---:|---:|']
    for name, d in diagnostics.items():
        g = d['official_geometry']
        lines.append(f"| {name} | {g['AU_total']:.6f} | {g['VA_total']:.6f} | {g['expression_total']:.6f} | {g['dynamic_FRDiv']:.6f} | {d['soft_geometry']['FRDiv']:.6f} |")
    lines += ['', '## Provenance and limits', '',
              'Full numerical diagnostics and paired session-bootstrap intervals are saved in the local `analysis.json`.',
              'This is single-seed VAL exploration, not a TEST score.',
              'The T1 teacher was a 600-step-per-model pilot with a quality cost versus T0; retaining T1 quality does not establish T0-level quality.',
              'Base commit plus exact training-file hashes are recorded because this experiment used worktree changes.', '']
    lines.append(f"- Teacher reference file SHA256: `{summary['teacher_reference_sha256']}`.")
    for name, p in matched['provenance'].items():
        lines += [f"- {name}: checkpoint `{checkpoint_hashes[name]}`, initial state `{p['initial_student_state_sha256']}`, source order `{p['observed_source_schedule_sha256']}`."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics': metrics, 'teacher_reproduction_delta': reproduction_delta}, indent=2))


if __name__ == '__main__':
    main()
