"""Generate the Phase 1.5 report/plots and a complete artifact hash inventory."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .phase15_audit import sha256_file


def session_bootstrap_difference(reference, other, group='overall', key='ES', seed=1503):
    """Other minus reference; bootstrap 20 sessions, not dependent clip pairs.

    Percentile intervals describe this selected validation set and one fixed
    shuffle/noise bank/checkpoint, not training-seed or permutation uncertainty.
    """
    grouped=defaultdict(list)
    for a,b in zip(reference['per_example'],other['per_example']):
        if a['clip_id']!=b['clip_id'] or a['session_id']!=b['session_id']:
            raise ValueError('paired records must be aligned')
        grouped[a['session_id']].append(b['channels'][group][key]-a['channels'][group][key])
    values=np.asarray([np.mean(grouped[s]) for s in sorted(grouped)])
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return dict(delta=float(values.mean()),ci95=[float(v) for v in np.quantile(means,[.025,.975])],
                session_count=len(values),replicates=10000,seed=seed)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('PHASE15_REPORT.md'))
    args=parser.parse_args()
    run=args.run; evaluation=run/'evaluation'
    summary=json.loads((evaluation/'summary.json').read_text())
    cases=summary['cases']
    real=json.loads((evaluation/'real_real_reference.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    trained=json.loads((run/'training_summary.json').read_text())
    get=lambda arm,k=32:cases[f'{arm}_correct_alpha1_K{k}']
    f=lambda x:f'{x:.8g}'
    comparisons={}
    for k in (32,64):
        reference=json.loads((evaluation/f'A1_correct_alpha1_K{k}.json').read_text())
        for name in ('shuffled','dataset_mean'):
            other=json.loads((evaluation/f'A1_{name}_alpha1_K{k}.json').read_text())
            comparisons[f'{name}_minus_correct_K{k}']={group:session_bootstrap_difference(reference,other,group)
                                                      for group in ('overall','AU','VA','expression')}
    (evaluation/'session_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    alphas=[0.,.25,.5,1.,1.5,2.]
    sweep=[cases[f'A1_correct_alpha{a:g}_K32'] for a in alphas]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key,label in [('cross_distance','Cross'),('self_distance','Self'),('ES','ES')]:
        axes[0,0].plot(alphas,[r['channels']['overall'][key] for r in sweep],marker='o',label=label)
    axes[0,0].set(title='A1 fixed sigma-scale sweep (K=32)',xlabel='alpha',ylabel='Normalized distance')
    for group in ('AU','VA','expression'):
        axes[0,1].plot(alphas,[r['channels'][group]['prediction_std'] for r in sweep],marker='o',label=group)
    axes[0,1].set(title='Sample spread by channel',xlabel='alpha',ylabel='Prediction std')
    labels=['Correct','Shuffled','Mean']; x=np.arange(3)
    for k,offset in [(32,-.12),(64,.12)]:
        values=[cases[f'A1_{name}_alpha1_K{k}']['channels']['overall']['ES'] for name in ('correct','shuffled','dataset_mean')]
        axes[1,0].plot(x+offset,values,marker='o',label=f'K={k}')
    axes[1,0].set(title='Same A1 checkpoint: prior assignment',xticks=x,xticklabels=labels,ylabel='ES (zoomed scale)')
    axes[1,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    groups=['AU','VA','expression'];x=np.arange(3)
    for offset,arm in [(-.27,'A0'),(-.09,'A_global'),(.09,'A1')]:
        axes[1,1].bar(x+offset,[get(arm)['channels'][g]['self_distance'] for g in groups],width=.18,label=arm)
    axes[1,1].bar(x+.27,[real['aggregate'][g]['distance']['mean'] for g in groups],width=.18,label='Real-real reference')
    axes[1,1].set(title='Generated spread vs descriptive real reference',xticks=x,xticklabels=groups,ylabel='Distance')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
    fig.savefig(evaluation/'attribution_curves.png',dpi=160)
    svg=evaluation/'attribution_curves.svg';fig.savefig(svg);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    k32_rows=[]
    for arm in ('A0','A_global','A1'):
        r=get(arm)['channels']['overall']
        k32_rows.append([arm,*[f(r[key]) for key in ('cross_distance','self_distance','ES','prediction_std')]])
    delta_rows=[]
    for k in (32,64):
        for name in ('correct','shuffled','dataset_mean'):
            r=cases[f'A1_{name}_alpha1_K{k}']['channels']['overall']
            ci=comparisons.get(f'{name}_minus_correct_K{k}',{}).get('overall')
            delta_rows.append([k,name,f(r['cross_distance']),f(r['self_distance']),f(r['ES']),
                               f(ci['delta']) if ci else '0',str([float(f(v)) for v in ci['ci95']]) if ci else '-'])
    channel_rows=[]
    for arm in ('A0','A_global','A1'):
        for group in ('AU','VA','expression'):
            r=get(arm)['channels'][group]
            channel_rows.append([arm,group,*[f(r[key]) for key in ('cross_distance','self_distance','ES','prediction_std','boundary_fraction','velocity_abs_mean','velocity_rms')],
                                 f(r['expression_entropy']) if group=='expression' else '-'])
    lines=['# HiRP Phase 1.5 attribution audit','',
'Base commit: `e03870fe01c8afdd0f633b8ee90acfb6da13c49b`. Core model/config files are unchanged. All three arms were initialized from scratch; no old checkpoint was loaded for training. Only the resulting audit checkpoints were loaded for read-only evaluation. No new loss, regularizer, bound, residual-scale change, group training, or official test.', '',
'## Interpretation', '',
'Within this short single-seed experiment, A1 has a much better Energy Score than A0/A_global, but correct input-prior assignment has no observable useful advantage over dataset-mean prior and only a tiny difference from a fixed within-session shuffle. The score benefit is dominated by stochastic spread rather than lower paired cross distance. This is evidence against claiming useful input-conditioned adaptation at this stage; it is not a theorem that such conditioning cannot help.', '',
'A_global learns only modest bias changes (sigma mean about 1.022), while A1 expands sigma to about 4.740. Same learning rate and step count do not equalize optimization speed between bias-only and context-amplified parameterizations. Therefore the between-arm comparison alone does not establish that global learned noise cannot help. The same-checkpoint mean/shuffle interventions provide the more direct attribution test. No condition-to-latent semantic meaning is claimed.', '',
'## Design and audit', '',
'- A0: mu=0, log_sigma=0 in an experiment-side hook; all parameters retained.',
'- A_global: prior forward pre-hook replaces context with zeros. Prior weights have gradients disabled for the forward, preventing both learning and AdamW decay; biases remain learnable. All original parameter objects remain. Weight equality after 64 steps is asserted.',
'- A1: unchanged conditional prior. Every arm contains 8,728,472 parameters and uses exactly identical initialization, 32 train clips, batch order, independent training noise, optimizer settings and steps; SHA-256 comparisons passed.',
'- Training: GPU 1, T=128, K=4, B=4, 64 steps/arm, AdamW lr=3e-4, weight_decay=0.01 (unchanged from Phase 1), seeds 123/456/789. No long training or added regularization.',
'- Validation: 80 fixed clips, 4 sampled per each of 20 sessions using seed 1501. Exact indices, IDs, sessions, crop offsets and lengths are in manifest.json. Equal session allocation intentionally does not estimate the original imbalanced clip-frequency distribution.',
'- Evaluation: common [1,64,32] Gaussian bank, seed 321, shared across inputs/arms/interventions. K=32 is its prefix; K=64 checks the three arms and correct/shuffled/mean. Sweep is evaluation-only alpha in [0,.25,.5,1,1.5,2], with z=mu+alpha*sigma*epsilon and no additional clamping.',
'- Shuffled prior: one fixed seed-1502 derangement inside each session, no self assignment; exact donor/recipient IDs saved. Mean prior: arithmetic mean mu and arithmetic mean sigma across all 80 validation inputs, no targets; independent of evaluation batching.',
'- Checkpoint file hashes and model-state hashes were unchanged after evaluation. Evaluation refuses mismatched data/normalization/checkpoint hashes. training_manifest.json is immutable; manifest.json adds final checkpoint hashes.',
'- 448 selected raw data files (4 per each of 112 train+validation clips) have individual SHA-256, plus mean_face.npy/std_face.npy. The canonical data-file list itself is hashed. Exact runtime source was archived in runtime_source.tar.gz and matches the original training/evaluation source hashes.', '',
'Random-crop preflight analytically enumerates all source-only starts 0..max(0,S-128); a start >= target_length would have zero overlap. Across all 1660 train and 571 val records, zero records have such a start. Total source/target lengths differ for 7 train and 8 val records. Shortest source: train=58, val=245. Cropping for these experiments remains deterministic center cropping.', '',
'## Main K=32 results', '',
table(['Arm','Cross','Self','ES','Prediction std'],k32_rows), '',
'## Same-checkpoint interventions', '',
'Positive delta = intervention ES minus correct ES (worse). Bootstrap intervals resample 20 sessions (10,000 resamples, seed 1503), preserving dependence among 4 clips in each session. They do not capture training-seed, noise-bank or shuffle-permutation uncertainty.', '',
table(['K','Prior','Cross','Self','ES','Delta ES','Session-bootstrap 95% interval'],delta_rows), '',
'## Sigma sweep (A1, K=32)', '',
table(['alpha','Cross','Self','ES','Prediction std'],[[a,*[f(r['channels']['overall'][key]) for key in ('cross_distance','self_distance','ES','prediction_std')]] for a,r in zip(alphas,sweep)]), '',
'At alpha=0 predictions are identical across samples; the self term is still 0.0001 because the unchanged norm includes eps=1e-8. That is a numerical floor, not actual diversity. Larger alpha lowers ES mainly via a larger self term while cross distance worsens; the sweep is not used to retrain or select a final model.', '',
'## Per-channel results (K=32)', '',
'All distances use RMS Euclidean norm over valid pair frames and the indicated channels, in stored channel units. No channel reweighting was added. Per-channel ES is evaluation-only. Per-recording metrics are macro-averaged. Boundary = within 0.01 of [0,1] endpoints for AU/expression or [-1,1] for VA. Entropy uses natural logs. Velocity uses first frame differences, without an assumed fps.', '',
table(['Arm','Channels','Cross','Self','ES','Pred std','Boundary fraction','Velocity abs','Velocity RMS','Expr entropy'],channel_rows), '',
'## Real-real descriptive reference', '',
'120 unordered within-session pairs among the selected 80 real listener trajectories. Distance uses the common valid crop length and relative frame positions; summary distances also compare each channel temporal mean/std. These are different speaker inputs, not ground-truth draws from the same conditional distribution. They never enter training, and are not a target spread that generation should be forced to match.', '',
table(['Channels','Real-real mean','p05','p50','p95','A1 generated self'],[[g,*[f(real['aggregate'][g]['distance'][key]) for key in ('mean','p05','p50','p95')],f(get('A1')['channels'][g]['self_distance'])] for g in ('overall','AU','VA','expression')]), '',
table(['Real channels','Boundary fraction','Velocity abs','Velocity RMS','Entropy'],[[g,*[f(real['aggregate'][g]['summaries'][key]['mean']) for key in ('boundary_fraction','velocity_abs_mean','velocity_rms')],f(real['aggregate'][g]['summaries']['expression_entropy']['mean']) if g=='expression' else '-'] for g in ('AU','VA','expression')]), '',
'Full real temporal mean/std, first-difference summaries, distribution mean/std/p05/p50/p95, per-recording values, pair distances and summary-vector distances are in evaluation/real_real_reference.json. Generated per-recording spread distributions are also saved in every case file.', '',
'Notable mismatch: generated VA sits at a boundary for about 50% of elements, versus about 0.093% in the selected real VA; A1 VA velocity RMS is also much higher than real VA. A1 expression entropy is much higher than real expression entropy. These are descriptive failures, not evidence of calibrated reactions.', '',
'## Prior clamp and residual saturation', '',
table(['Arm','sigma mean','sigma std','sigma min','sigma max','Lower clamp','Upper clamp','raw abs mean','residual abs mean','tanh saturation'],[[arm,*[f(get(arm)['prior'][key]) for key in ('sigma_mean','sigma_std','sigma_min','sigma_max','log_sigma_lower_clamp_fraction','log_sigma_upper_clamp_fraction')],*[f(get(arm)['residual'][key]) for key in ('stochastic_raw_abs_mean','stochastic_residual_abs_mean','tanh_saturation_fraction')]] for arm in ('A0','A_global','A1')]), '',
'Tanh saturation threshold is abs(tanh(stochastic_raw))>=0.99, measured only on source-valid frames. Prior clamp statistics refer to the base prior before alpha scaling. No upper bound or residual scale was changed.', '',
'## K=64 stability', '',
table(['Arm','K32 ES','K64 ES','Difference'],[[arm,f(get(arm,32)['channels']['overall']['ES']),f(get(arm,64)['channels']['overall']['ES']),f(get(arm,64)['channels']['overall']['ES']-get(arm,32)['channels']['overall']['ES'])] for arm in ('A0','A_global','A1')]), '',
'![Attribution curves](runs/phase15/evaluation/attribution_curves.png)', '',
'## Tests and artifacts', '',
'Complete CPU results are in runs/phase15/pytest_results.txt. Tests cover bias-only global updates with AdamW, parameter preservation and exception cleanup, K<2 errors, reproducible stratification/derangement, arithmetic mean sigma, eval-only overrides and alpha=0, channel ES equivalence/masking, entropy/boundaries/velocity, real-real session restrictions, exact random-crop risk, raw-data hash mismatch detection, and checkpoint roundtrip. There is one PyTorch TypedStorage deprecation warning.', '',
'Runtime checkpoint paths (local, intentionally Git-ignored):', '',
table(['Arm','Checkpoint','SHA-256'],[[arm,v['path'],v['sha256']] for arm,v in manifest['checkpoints'].items()]), '',
'Other artifacts: training_manifest.json / manifest.json; training_summary.json; A0/A_global/A1_training.jsonl; random_crop_audit.json; runtime_source.tar.gz; evaluation/manifest.json, common_noise_K64.npy, *_prior_cache.npz, 15 per-case JSON files, summary.json, session_bootstrap.json, real_real_reference.json, attribution_curves.png/svg. artifact_inventory.json lists every file with bytes and SHA-256 except the inventory itself. No raw REACT data were copied.', '',
'All requested Phase 1.5 work completed. Limits: one short training seed, one fixed within-session shuffle, center crops of length 128, 80 stratified examples, and different optimization dynamics for bias-only versus conditional parameterization. No claim of useful input-prior adaptation is supported here. Checkpoints are retained for further read-only evaluation. No additional training was performed after attribution.', '',
'Replay evaluation into a NEW directory (does not modify checkpoint files):', '',
'```bash',
'.venv/bin/python -m hirp.evaluate_phase15 --run runs/phase15 --output runs/phase15/evaluation_repeat --device cuda:1',
'```', '']
    args.report.write_text('\n'.join(lines))
    inventory=[]
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name!='artifact_inventory.json':
            inventory.append(dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=sha256_file(path)))
    (run/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2))
    print('REPORT',str(args.report),'ARTIFACTS',len(inventory),flush=True)
    print('PAIRED_BOOTSTRAP',json.dumps({name:v['overall'] for name,v in comparisons.items()}),flush=True)


if __name__=='__main__':main()
