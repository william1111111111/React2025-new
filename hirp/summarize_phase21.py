"""Reproducible report and paired-session bootstrap; no model execution."""
import json
from pathlib import Path
import numpy as np
from .train_phase21 import write


def main():
    run=Path('runs/phase21/conditional_score_v0')
    def read(name):return json.loads((run/name).read_text())
    arms=('C0','C1','C2'); cases={a:read(f'evaluation/{a}.json') for a in arms}
    rng=np.random.default_rng(2106); indices=rng.integers(0,20,(10000,20)); bootstrap={}
    def interval(x):
        x=np.asarray(x); lo,hi=np.quantile(x[indices].mean(1),[.025,.975])
        return dict(mean=float(x.mean()),lower=float(lo),upper=float(hi))
    for a in arms:
        rows=cases[a]['per_session']
        bootstrap[a+'_shuffle_gap']=interval([r['conditional_shuffled']['loss']-r['conditional_correct_common']['loss'] for r in rows])
    for b in ('C0','C1'):
        assert [r['session_id'] for r in cases['C2']['per_session']]==[r['session_id'] for r in cases[b]['per_session']]
        for metric in ('conditional','marginal'):
            bootstrap[f'C2_minus_{b}_{metric}']=interval([r[metric]['loss']-s[metric]['loss'] for r,s in zip(cases['C2']['per_session'],cases[b]['per_session'])])
    write(run/'evaluation/session_bootstrap.json',dict(seed=2106,resamples=10000,unit='20 paired sessions',contrasts=bootstrap))
    lines=['# HiRP Phase 2.1: conditional distribution supervision',
        '', 'Base / local HEAD at resumption: `8fbc5febddb14b0b54d27633a3edb95227cc9834`; branch `agent/hirp-phase21-conditional-score`. No later commits existed. Existing uncommitted work was retained. See resumption_audit.json and code_changes.patch. No push.',
        '', '## Findings', '',
        'The pre-norm output path restores measurable stochastic trainability in this bounded experiment. C0 has the best held-out paired trajectory ES. C2 improves group descriptor ES relative to C0 while worsening conditional ES; compared with C1, C2 improves both conditional and marginal ES. All three correct-input scores beat the fixed within-session shuffle. This supports input correspondence sensitivity, not recovery of each full conditional distribution.',
        '', 'Self distance is a legitimate term of ES. A self-driven improvement is not by itself gaming; neither is improved marginal ES proof of conditional calibration. AU/expression spread remains small and expression entropy remains high relative to real references. No complete mechanism or MARS performance claim is established.',
        '', '## Objectives and controls','',
        '| Experiment | Paired term | Distribution term |','|---|---|---|',
        '| Historical B2/B3/B4 | RMS(G(X,0),Y) | paired descriptor / per-input population / cross-bank group ES |',
        '| C0 | trajectory ES over every stochastic sample | none |',
        '| C1 | same trajectory ES | 0.1 × unchanged b3_score |',
        '| C2 | same trajectory ES | 0.1 × unchanged b4_score |',
        '', 'G(X,0) is diagnostic only, neither an expectation nor a median. Trajectory scores use pair_lengths and 25 unit channel scales; generated descriptors use source_lengths and the frozen train-only 75-coordinate scaler. Targets/references do not enter forward. A0 prior is frozen; encoder/decoder remain conditional on X.',
        '', 'All arms: scratch seed 123, T=128, K_train=4, B=4, 128 steps, AdamW lr=1e-4, weight decay .01, float32, pre-norm with default projection initialization, 8,729,496 parameters. FiLM small nonzero initialization and output domains/residual formula are preserved. The optional small projection initialization was implemented/tested but not used for the pilot.',
        '', 'The original Phase 2 schedule and independent noise were hash-verified and reused. Four source occurrences are IID with replacement, split into two banks; references are unique within a step and train-only. C1/C2 consumed identical references. C0 has zero reference lookups in its loss/step path; the shared experiment preparation loads a reference cache for C1/C2. This is not three separately isolated data-loading processes.',
        '', 'Lambda was selected from [.01,.03,.1,.3,1] on 8 TRAIN batches × 2 arms, nearest log-distance to a median weighted group/conditional gradient ratio of .5. Selected .1; actual initial median ratio .771551. This is a reasonable fixed starting point, not optimality or balance evidence. Gradient norms/cosines recur every 16 steps; no dynamic controller.',
        '', '## Output-path evidence','',
        '32-step old/pre-norm smoke uses the same conditional objective, projection/backbone initialization and learning rate. The locked TRAIN-only viability rule rejected old-head and accepted pre-norm. Historical B2/B3/B4 checkpoints were only probed read-only; no structural checkpoint migration.',
        '', '| Probe at final step | Residual saturation | Median tanh derivative | AU noise VJP | VA noise VJP | Expression noise VJP |', '|---|---:|---:|---:|---:|---:|']
    for a in ('old_head','pre_norm_head',*arms):
        p=read(a+'_path_diagnostics.json')[-1]
        lines.append('| '+a+' | '+' | '.join(f'{x:.7g}' for x in [p['residual']['tanh_saturation_fraction'],p['residual']['derivative']['abs_quantiles']['p50'],*[p['noise_jacobian'][g]['vjp_norm'] for g in ('AU','VA','expression')]])+' |')
    lines+=['','Full *_path_diagnostics.json records encoder H, decoder hidden, base/stochastic logits and final raw: valid-frame mean/std/absolute quantiles and conditional gradient norms, plus boundaries, sample std and derivative quantiles. Noise values are seeded normalized random-output VJPs, not full Jacobian norms. Prior parameters remained unchanged. These fixed TRAIN probes are not whole-dataset saturation estimates.', '', '| Historical arm | Base VA abs p95 | Final VA abs p95 | Residual saturation |','|---|---:|---:|---:|']
    for a,v in read('historical_output_path.json').items():
        p=v['diagnostic'];lines.append(f"| {a} | {p['paths']['base_logits']['VA']['abs_quantiles']['p95']:.6g} | {p['paths']['final_raw']['VA']['abs_quantiles']['p95']:.6g} | {p['residual']['tanh_saturation_fraction']:.6g} |")
    lines+=['','Base and stochastic paths must both be considered: bounded stochastic residuals cannot undo arbitrarily large base logits. Pre-norm normalizes both projections; the evidence does not isolate which normalization is individually necessary.', '', '| Arm | Weighted group/conditional gradient ratio min–max | Gradient cosine min–max |', '|---|---:|---:|']
    for a in ('C1','C2'):
        rows=[json.loads(s)['component_gradients'] for s in (run/(a+'_training.jsonl')).read_text().splitlines()];rows=[r for r in rows if r]
        r=[v['weighted_group_over_conditional'] for v in rows];c=[v['cosine'] for v in rows]
        lines.append(f'| {a} | {min(r):.6g}–{max(r):.6g} | {min(c):.6g}–{max(c):.6g} |')
    lines+=['','## Held-out fixed-80 pilot', '', 'K=32 common Gaussian bank; 4 clips × 20 sessions, all 571 unique VAL listener references. Session-macro averages; common-mask correct/shuffled comparisons. Marginal self excludes equal noise indices for every source pair. No official metrics or hidden test.', '', '| Arm | Paired ES | Cross | Self | Shuffled ES | Shuffle gap | Group ES | Central RMS (aux) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in arms:
        s=cases[a]['aggregate'];vals=[s['conditional']['loss'],s['conditional']['cross_distance'],s['conditional']['self_distance'],s['conditional_shuffled']['loss'],s['conditional_shuffle_gap']['difference'],s['marginal']['loss'],s['central']['overall']]
        lines.append('| '+a+' | '+' | '.join(f'{v:.7f}' for v in vals)+' |')
    lines+=['','| Arm | Marginal cross | Marginal self | Within-input descriptor self | Between-input descriptor distance |','|---|---:|---:|---:|---:|']
    for a in arms:
        s=cases[a]['aggregate']; vals=[s['marginal']['cross'],s['marginal']['self'],*s['spread'].values()]
        lines.append('| '+a+' | '+' | '.join(f'{v:.7f}' for v in vals)+' |')
    lines+=['', 'For four fixed sources, marginal self = within/4 + 3×between/4, excluding shared noise indices. Descriptor units differ from raw trajectory units.', '', '| Contrast | Mean | Session bootstrap 95% interval |','|---|---:|---|']
    for name,v in bootstrap.items():lines.append(f"| {name} | {v['mean']:.7f} | [{v['lower']:.7f}, {v['upper']:.7f}] |")
    lines+=['','Bootstrap uses 10,000 paired session resamples, seed 2106. It excludes training-seed, noise-bank and shuffle-permutation uncertainty.', '', '| Arm/channel | Sample std | Boundary fraction | Velocity RMS | Expression entropy |','|---|---:|---:|---:|---:|']
    for a in arms:
        for g in ('AU','VA','expression'):
            v=cases[a]['aggregate']['channels'][g]
            lines.append(f"| {a}/{g} | {v['prediction_std']:.7g} | {v['boundary_fraction']:.7g} | {v['velocity_rms']:.7g} | {v.get('expression_entropy','—')} |")
    lines+=['','Real reference distributions/crops are saved in evaluation/real_reference.json; per-input and session metrics plus held-out path probes in evaluation/C*.json. References describe different X, not repeated conditional draws at one X.', '', '## Toy recovery','', 'Same-session Bernoulli probabilities [.1,.9], 8192 sampled labels per input, seed 2103. A two-input linear sigmoid network optimizes exact expected ES or the full sampled-label empirical objective. Initializations [.5,.5], [.2,.4], [.8,.3]; 500 steps. This validates the objective connection, not REACT generalization.', '', '| Initialization | Objective | Arm | Recovered probabilities | Conditional MAE | Marginal error |','|---|---|---|---|---:|---:|']
    for r in read('toy_recovery.json')['rows']:
        lines.append(f"| {r['initial']} | {r['kind']} | {r['arm']} | {str([round(v,7) for v in r['probabilities']])} | {r['conditional_probability_mae']:.7g} | {r['marginal_probability_error']:.7g} |")
    lines+=['','Analytic C0/C2 optimum is [.1,.9]; C1 optimum is [.1363636,.8636364] at lambda=.1. The original optimum-initialized compatibility toy is unchanged.', '', '## Validation, provenance and remaining work','', 'Actual earlier stage commands and failures: commands.jsonl. Earlier full test result: 101 passed, 1 xfailed, 1 warning in pytest_before_pilot.txt. Resumed sandbox tests: 98 passed, 3 CUDA skipped, 1 xfailed, 1 warning. GPU-enabled repeat: 101 passed, 1 xfailed, 1 warning in 13.74s (pytest_resumed_gpu.txt). CPU BF16 fused Transformer eval/no_grad on PyTorch 2.1 is explicitly xfailed; pilot/evaluation use float32, and ES distances remain float32 under AMP.', '', 'Resumed evaluation command: `.venv/bin/python -m hirp.evaluate_phase21 --run runs/phase21/conditional_score_v0 --device cuda:2`. Full GPU test command: `.venv/bin/python -m pytest hirp/tests -q`. Report/bootstrap command: `.venv/bin/python -m hirp.summarize_phase21`.', '', '83 historical artifact hashes match the preparation snapshot. Runtime smoke/training archives and checkpoint hashes are retained. No old code/report/data/checkpoint/official metric changes. Development failure logs are preserved. Checkpoints are local Git-ignored files.', '', 'Not run: long training, multi-seed pilot, K=64 sensitivity, multiple shuffles/noise banks, hidden/official evaluation, small-projection-init training comparison, separate base-only/stochastic-only normalization ablations, conditional-prior learning, or push. Neither learning rate nor lambda is claimed optimal. The existing bounded pilot was reused, not repeated after validation.']
    (run/'PHASE21_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
