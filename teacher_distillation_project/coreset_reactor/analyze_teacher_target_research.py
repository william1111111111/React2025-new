"""Aggregate all fixed-teacher diagnostics and both predeclared pilot arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coreset_reactor.analyze_t1_student_refinement import paired_deltas
from coreset_reactor.teacher_cache import sha256_file


def means(rows):
    return {'contexts': len(rows),
            'FRC': float(np.mean([sum(r['frc']) for r in rows])),
            'exact_FRD': float(np.mean([r['exact_FRD'] for r in rows])),
            'FRDiv': float(np.mean([r['FRDiv'] for r in rows]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--convergence-root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    root, old = args.root, args.convergence_root
    fit = json.loads((root/'train_fit_full.json').read_text())
    identity = json.loads((root/'identity_self_match.json').read_text())
    targets = json.loads((root/'manifests/T2R/summary.json').read_text())
    val_representation = json.loads((root/'val_representation_diagnostic.json').read_text())
    pilot = {name: json.loads((root/folder/'summary.json').read_text())
             for name, folder in [('T2R', 'pilot600'), ('T1P', 'processed_pilot600')]}
    for run in pilot.values():
        assert run['matched_training_checks'] == run['matched_evaluation_checks'] == 'passed'
    assert fit['contexts'] == identity['contexts'] == 1660
    assert {r['clip_id'] for r in fit['rows']} == {r['clip_id'] for r in identity['rows']}
    paths = {'T0': old/'eval/T0_s000600_val571_full.json',
             'T1': old/'eval/T1_s000600_val571_full.json',
             'T2R': root/'pilot600/val571_full.json',
             'T1P': root/'processed_pilot600/val571_full.json'}
    results = {k: json.loads(v.read_text()) for k, v in paths.items()}
    reference = [(r['clip_id'], r['target_cache_key']) for r in results['T0']['rows']]
    for result in results.values():
        assert result['metrics']['contexts'] == 571
        assert [(r['clip_id'], r['target_cache_key']) for r in result['rows']] == reference
    train_sessions = {r['clip_id'].split('/')[1] for r in fit['rows']}
    strata = {name: {group: means([r for r in result['rows'] if
                                  (r['clip_id'].split('/')[1] in train_sessions) == seen])
                     for group, seen in [('seen_session', True), ('unseen_session', False)]}
              for name, result in results.items()}
    delta = {'T2R_vs_T0': paired_deltas(results['T2R']['rows'], results['T0']['rows']),
             'T1P_vs_T1': paired_deltas(results['T1P']['rows'], results['T1']['rows'])}
    matching = {}
    for label in fit['summary']:
        values = [r['arms'][label] for r in fit['rows']]
        matching[label] = {}
        for suffix, fixed, best in [
            ('training_aligned', 'fixed_training_aligned_rounded', 'all_session_training_aligned_rounded'),
            ('official_processed', 'fixed_official_processed', 'all_session_official_processed')]:
            a, b = np.array([r[fixed] for r in values]), np.array([r[best] for r in values])
            matching[label][suffix+'_fixed_is_best_fraction'] = float(np.mean(np.isclose(a, b, atol=1e-6, rtol=0)))
            matching[label][suffix+'_specialist_fixed_is_best_fraction'] = float(np.mean(np.isclose(a[:, 1:], b[:, 1:], atol=1e-6, rtol=0)))
    identity_comparison = {arm: {
        'same_training_GT_self_score': identity['summary'][arm],
        'same_GT_after_official_processing': fit['summary'][f'{arm}_600']['exact_target_to_same_processed_identity'],
        'best_all_session_processed_GT': fit['summary'][f'{arm}_600']['exact_target_to_all_processed_GT'],
    } for arm in ('T0', 'T1')}
    summary = {'metrics': {k: v['metrics'] for k, v in results.items()}, 'pilot_checks': pilot,
               'VAL_representation_diagnostic': val_representation['summary'],
               'train_fit_summary': fit['summary'], 'identity_comparison': identity_comparison,
               'fixed_match_fraction': matching, 'session_strata': strata, 'matched_deltas': delta,
               'target_usage': targets['target_usage'], 'T2R_TRAIN_target_FRDiv': targets['TRAIN_target_FRDiv'],
               'diagnostic_target_length_mismatch_fraction': sum(r['target_length_mismatch_count'] for r in fit['rows'])/sum(r['gt_count'] for r in fit['rows']),
               'analysis_sha256': sha256_file(Path(__file__)),
               'evaluation_sha256': {k: sha256_file(v) for k, v in paths.items()},
               'diagnostic_sha256': sha256_file(root/'train_fit_full.json'),
               'identity_sha256': sha256_file(root/'identity_self_match.json'),
               'limitations': ['one seed and repeatedly inspected VAL; not an independent TEST result',
                               'TRAIN diagnostic excludes reverse roles, while training includes both roles',
                               'target processing changes temporal alignment and values together',
                               'T2R bundles density support and reuse cost; component attribution requires ablation',
                               'T1P GT/length cache seeds differ from per-context eval seeds',
                               'session strata also differ in GT-pool size',
                               'no student retrained']}
    (root/'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Teacher target research: TRAIN fit, supported GTs, and target representation', '',
             'REACT2025; full VAL571; all-session opposite-role GT; AU rounding; exact rolling FRD. This uses the established all-session extension, not the official random-10 GT sampling contract.', '',
             '| Arm, 600 updates/model | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar |',
             '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        m = result['metrics']
        lines.append(f"| {name} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {m['FRVar']:.6f} |")
    lines += ['', 'T0: random fixed paired+9; T1: pure Max-FRDiv; T2R: supported prototypes/reuse penalty; T1P: unchanged T1 target IDs, official-processed TRAIN specialist targets.', '',
              '## Predeclared guard outcomes', '']
    for name in pilot:
        lines.append(f"- {name}: strict guard {'PASS' if pilot[name]['strict_guard_pass'] else 'FAIL'} against {'T0' if name == 'T2R' else 'T1'}.")
    lines += ['', 'All arms retain exactly ten outputs, the same architecture and 600-step budget. A higher FRC with lost diversity is not a strict win.', '',
              '| Contrast | ΔFRC [session bootstrap95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |',
              '|---|---|---|---|']
    for label, changes in delta.items():
        cells = []
        for metric in ('FRC', 'exact_FRD', 'FRDiv'):
            d = changes[metric]
            lo, hi = d['session_bootstrap_CI95']
            cells.append(f"{d['mean_delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        lines.append(f'| {label} | '+ ' | '.join(cells)+' |')
    lines += ['', '## Full TRAIN fixed-target diagnosis', '',
              'All 1660 TRAIN speaker inputs, deterministic center source crops, eval-mode teachers. Values are sums of ten per-prediction CCC contributions, then context means. TRAIN fit is not held-out quality.', '',
              '| Arm / steps | Fixed raw soft | Fixed raw rounded | All-session raw rounded | Fixed processed | All-session processed | VAL all-session processed |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, values in fit['summary'].items():
        arm, step = label.split('_')
        val = json.loads((old/f'eval/{arm}_s{int(step):06d}_val571_full.json').read_text())['metrics']['FRC']
        keys = ('fixed_training_aligned_soft', 'fixed_training_aligned_rounded',
                'all_session_training_aligned_rounded', 'fixed_official_processed', 'all_session_official_processed')
        lines.append(f'| {label} | '+' | '.join(f'{values[k]:.6f}' for k in keys)+f' | {val:.6f} |')
    lines += ['', 'The fixed TRAIN GT is explicitly in the TRAIN candidate pool. Best-GT FRC >= fixed-GT FRC is asserted for every prediction under each representation. These identities are not in VAL; this is expected split separation.', '',
              'T1 fits its fixed raw targets increasingly well from 600 to 2400 updates, but held-out processed FRC does not improve monotonically. This rules out a simple claim that the fixed mapping cannot be fitted; it does not establish which part of the remaining gap is caused by representation, memorization, or other generalization effects.', '',
              '## Exact-GT identity control', '',
              '| Target family | GT vs own training representation | Same identity after official processing | Best all-session processed identity |',
              '|---|---:|---:|---:|']
    for arm, values in identity_comparison.items():
        lines.append(f'| {arm} | '+' | '.join(f'{x:.6f}' for x in values.values())+' |')
    lines += ['', 'Here the predictor is replaced by the actual training GT. The comparison isolates the combined temporal/representation change from model fitting and split generalization. Constant channels have CCC0, so a self-match need not equal10. A low score is not automatically proof that any evaluator implementation is incorrect.', '',
              '| Arm / steps | Fixed GT is best: raw | Fixed GT is best: processed |', '|---|---:|---:|']
    for label, values in matching.items():
        lines.append(f"| {label} | {values['training_aligned_fixed_is_best_fraction']:.4f} | {values['official_processed_fixed_is_best_fraction']:.4f} |")
    lines += ['', '## GT reuse and session strata', '',
              '| Family | Distinct specialist GT/rank (min–max) | Effective GT/rank (min–max) |', '|---|---:|---:|']
    for family, rows in targets['target_usage'].items():
        unique = [r['distinct_GT'] for r in rows[1:]]
        effective = [r['effective_GT'] for r in rows[1:]]
        lines.append(f'| {family} | {min(unique)}–{max(unique)} | {min(effective):.1f}–{max(effective):.1f} |')
    lines += ['', f"T2R TRAIN target-set FRDiv: {targets['TRAIN_target_FRDiv']:.6f}.", '',
              '| Arm | Session stratum | Contexts | FRC | exact FRD | FRDiv |', '|---|---|---:|---:|---:|---:|']
    for name, groups in strata.items():
        for group, m in groups.items():
            lines.append(f"| {name} | {group} | {m['contexts']} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} |")
    lines += ['', 'The unseen sessions are session2/session10/session18. GT-pool sizes differ across strata, so this is descriptive, not an isolated causal test of session novelty.', '',
              '## Post-hoc VAL representation check', '',
              'The same 571 predictions are rescored against relative-time raw GT and native paired raw GT. These are diagnostic scores, not a replacement for the official-processed protocol or new selection criteria.', '',
              '| Arm | All-session raw representation FRC | Native paired raw FRC | Official-processed all-session FRC |',
              '|---|---:|---:|---:|']
    for arm, values in val_representation['summary'].items():
        lines.append(f'| {arm} | '+' | '.join(f'{x:.6f}' for x in values.values())+' |')
    lines += ['',
              '## Interpretation and next decision', '',
              'Neither pilot is promoted: both fail their predeclared zero-slack guards. T2R improves FRC but does not retain the high-diversity T1 operating point, and its exact FRD is worse than T0. This does not isolate whether density support or reduced reuse caused the FRC gain.', '',
              'T1P improves official-processed FRC but reduces raw-target VAL FRC and slightly reduces FRDiv while worsening exact FRD. Thus matching the target representation alone is not a demonstrated improvement in original-trajectory fidelity. Do not replace the teacher bank based on its FRC alone.', '',
              'A fixed TRAIN identity is eligible in the TRAIN all-session search; it need not be the best-scoring identity after processing. TRAIN identities are not inserted into VAL. Both optimization/generalization and target representation must be considered; these diagnostics do not prove that the evaluator is wrong or that reuse alone caused the T1 gap.', '',
              'The next bounded experiment, if pursued, should retain the T1 GT identities and raw-trajectory supervision, adding only a weak processed-target auxiliary term rather than replacing the target. Keep source schedule, optimizer updates, and paired anchor fixed; preregister processed-FRC, exact-FRD, FRDiv and raw-target diagnostic guards. This is a proposal, not a completed or known-successful method. No additional teacher or student training was launched after these two pilots.', '',
              '## Compute and provenance', '',
              'Each arm has ten 3,720,129-parameter teachers (37,201,290 total). q0 is reused from its exact matched baseline; nine new models ×600 updates =5400 new optimizer steps per arm. Including q0, each full ensemble budget is 6000 updates. Batch32, BF16, AdamW1e-4, the original 50-epoch cosine prefix, loss, source schedule and initial state remain matched.', '',
              '| Arm | Mean training updates/s/model | Inference ms/context | Cache hits |', '|---|---:|---:|---:|']
    for name, directory in [('T2R', root/'pilot600'), ('T1P', root/'processed_pilot600')]:
        audits = [json.loads((directory/f'rank_{r:02d}/conditional-step000600-seed1.json').read_text()) for r in range(1, 10)]
        speed = results[name]['speed']
        lines.append(f"| {name} | {np.mean([600/a['elapsed_training_seconds'] for a in audits]):.3f} | {1000*speed['inference_latency_s_mean']:.2f} | {speed['target_cache_hits']}/571 |")
    lines += ['', 'Pool/output size10; one exact temporal target per source per teacher; selector overhead0. Concurrent GPU work makes latency/throughput diagnostic, not a clean benchmark. T1P also requires a one-time TRAIN VAE preprocessing cache; it does not run that processor at inference.', '',
              'T1P caches 5240 TRAIN GT/length combinations. Preprocessing wall time was not separately recorded, so optimizer-budget equality must not be interpreted as equality of total preparation cost.', '',
              'Initial parameter and actual source/crop hashes passed. All VAL target-cache keys and input order match their references. Checkpoint/data identities and per-context outputs stay local. Aggregate conclusions are single-seed and VAL-selected. Existing student recipes and weights remain unchanged.', '',
              'See [the predeclared protocol](TEACHER_TARGET_RESEARCH_PROTOCOL.md).']
    provenance = json.loads((root/'pilot600/provenance.json').read_text())
    lines += ['', f"Base commit: `{provenance['base_git_sha']}`. The pilots also record hashes of the uncommitted source files; the commit alone does not identify this research implementation.", '',
              f"Analysis source SHA256: `{summary['analysis_sha256']}`.", '',
              f"Merged TRAIN diagnostic SHA256: `{summary['diagnostic_sha256']}`."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics': summary['metrics'], 'identity_comparison': identity_comparison,
                      'guards': {k: v['strict_guard_pass'] for k, v in pilot.items()}}, indent=2))


if __name__ == '__main__':
    main()
