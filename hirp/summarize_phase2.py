"""Phase 2 evidence report; no parameter updates or extra experiments."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from .phase15_audit import sha256_file


def bootstrap(values,seed=2004):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    sampled=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(sampled,[.025,.975]).tolist(),sessions=len(values),resamples=10000,seed=seed)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('runs/phase2'))
    args=parser.parse_args();run=args.run
    manifest=json.loads((run/'training_manifest.json').read_text());trained=json.loads((run/'training_summary.json').read_text())
    preflight=json.loads((run/'gradient_preflight.json').read_text());toy=json.loads((run/'toy_hierarchy.json').read_text())
    if trained.get('stopped_before_training'):
        (run/'PHASE2_REPORT.md').write_text('# Phase 2 stopped\n\nThe pretraining gradient gate failed. No arm was trained; lambda was not changed. See gradient_preflight.json.\n')
        return
    evaluations={arm:json.loads((run/'evaluation'/f'{arm}.json').read_text()) for arm in ('B2','B3','B4')}
    aggregate={arm:value['aggregate'] for arm,value in evaluations.items()}
    real=json.loads((run/'evaluation/real_reference.json').read_text())
    real_a=real['aggregate'];f=lambda value:f'{value:.8g}'
    ses={arm:{r['session_id']:r for r in value['per_session']} for arm,value in evaluations.items()}
    sessions=sorted(ses['B4']);comparisons={}
    for baseline in ('B2','B3'):
        comparisons[f'B4_minus_{baseline}_marginal_ES']=bootstrap([ses['B4'][s]['marginal']['loss']-ses[baseline][s]['marginal']['loss'] for s in sessions])
        comparisons[f'B4_minus_{baseline}_central_RMS']=bootstrap([ses['B4'][s]['central_correct']['overall']-ses[baseline][s]['central_correct']['overall'] for s in sessions])
        for label,correct,shuffled,key in [('central','central_correct','central_shuffled','overall'),
                                          ('paired_descriptor','paired_descriptor_correct','paired_descriptor_shuffled','loss')]:
            differences=[]
            for session in sessions:
                a,b=ses['B4'][session],ses[baseline][session]
                differences.append((a[shuffled][key]-a[correct][key])-(b[shuffled][key]-b[correct][key]))
            comparisons[f'B4_minus_{baseline}_{label}_shuffle_gap']=bootstrap(differences)
    (run/'evaluation/session_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    decomposition={}
    for arm,r in aggregate.items():
        within=r['per_input_population']['self'];mixture=r['session_marginal']['self']
        decomposition[arm]=dict(within_input_descriptor_self=within,session_mixture_descriptor_self=mixture,
                                between_input_descriptor_distance=(4*mixture-within)/3,
                                identity='mixture_self = 1/4 within_input_self + 3/4 between_input_self; distinct noise indices')
    (run/'evaluation/spread_decomposition.json').write_text(json.dumps(decomposition,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['B2','B3','B4'];x=np.arange(3)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for offset,key,label in [(-.25,'cross','Cross'),(0,'self','Self'),(.25,'loss','ES')]:
        axes[0,0].bar(x+offset,[aggregate[a]['session_marginal'][key] for a in arms],width=.25,label=label)
    axes[0,0].set(title='Session marginal fit (normalized descriptor)',xticks=x,xticklabels=arms,ylabel='Distance')
    for offset,key,label in [(-.18,'central_conditionality','Central RMS'),(.18,'paired_descriptor_conditionality','Paired descriptor ES')]:
        axes[0,1].bar(x+offset,[aggregate[a][key]['ratio'] for a in arms],width=.36,label=label)
    axes[0,1].axhline(1,color='black',linewidth=.8)
    axes[0,1].set(title='Input shuffle / correct score (zoomed axis)',xticks=x,xticklabels=arms,ylabel='Ratio',ylim=(.99,1.06))
    for offset,key,label in [(-.18,'within_input_descriptor_self','Within input'),(.18,'between_input_descriptor_distance','Between inputs')]:
        axes[1,0].bar(x+offset,[decomposition[a][key] for a in arms],width=.36,label=label)
    axes[1,0].set(title='Where does descriptor spread come from?',xticks=x,xticklabels=arms,ylabel='Distance')
    for arm in arms:
        rows=[json.loads(line) for line in (run/f'{arm}_training.jsonl').read_text().splitlines()]
        smoothed=np.convolve([r['loss'] for r in rows],np.ones(8)/8,mode='valid')
        axes[1,1].plot(np.arange(8,len(rows)+1),smoothed,label=arm)
    axes[1,1].set(title='Training total loss (8-step moving mean)',xlabel='Step',ylabel='Pair + descriptor distribution loss')
    for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
    fig.savefig(run/'evaluation/phase2_curves.png',dpi=160)
    svg=run/'evaluation/phase2_curves.svg';fig.savefig(svg);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    schedule=json.loads((run/'schedule.json').read_text());counts=Counter(r['session_id'] for r in schedule['records'])
    sampling=dict(source_occurrences=512,unique_sources=len(manifest['source_crops']),reference_occurrences=1024,
                  unique_sampled_references=len(manifest['reference_crops']),
                  steps_with_repeated_source=sum(len(set(r['source_indices']))<4 for r in schedule['records']),
                  session_probabilities={s:info['source_count']/manifest['train_pool']['source_population'] for s,info in manifest['train_pool']['sessions'].items()},
                  observed_session_counts=dict(sorted(counts.items())),hashes=schedule['hashes'])
    (run/'sampling_audit.json').write_text(json.dumps(sampling,indent=2))
    lines=['# Phase 2: Hierarchical Distribution Supervision — bounded pilot','',
'Branch: `agent/hirp-phase2-hierarchy`. Base commit: `c2d92c9df528ac42850b9a874005ff305d40473f`. Final local commit is supplied in the completion message (a commit cannot contain its own hash). No push. Core hirp/model files, model config, Phase 1/1.5 code/reports/artifacts are unchanged.', '',
'## Answer to Q1–Q4', '',
'**Q1 — Marginal fit:** B4 has lower validation session-marginal descriptor ES than B2/B3. However its cross distance is WORSE, and the gain comes from a much larger between-input self term. This supports an effect of the hierarchy connection on the supervised statistic, not calibrated population matching.', '',
'**Q2 — Correspondence:** B4 has a better overall central paired RMS and larger central input-shuffle gap than B3 in point estimates. Its AU and expression paired distances are worse; the overall gain is driven by VA. Paired descriptor ES is substantially worse and its shuffle gap is tiny. Session-bootstrap uncertainty is reported below; useful correspondence preservation is not established robustly.', '',
'**Q3 — Diversity shape:** B4 expression entropy is closer to the real-reference summary, and VA velocity becomes nonzero and closer in RMS. But VA boundary occupancy remains approximately 96.4% (real about 0.030%), and conditional AU/expression stochastic spread nearly collapses. The shape failure is not repaired.', '',
'**Q4 — Hierarchy versus variance:** B4 has tiny within-input stochastic spread and much larger BETWEEN-input descriptor distances. It is not winning by more within-input noise. Yet its ES gain is driven by larger aggregate marginal spread with worse cross fit; this does not exclude a variance-driven or extreme-output explanation. This pilot does NOT establish the intended complete mechanism.', '',
'## Implemented connection and fixed boundaries', '',
'New source: session_data.py, group_features.py, group_scores.py, train_phase2.py, evaluate_phase2.py, summarize_phase2.py, tests/test_phase2.py. Root .gitignore only adds the local Phase 2 checkpoint directory. All three models retain 8,728,472 parameters, use prior_mode=A0 and have unchanged ConditionalPrior parameters after training. No conditional Gaussian tuning or prohibited model/loss was introduced.', '',
'Every step uses two gradient-enabled model.forward calls: zero noise, K=1 central prediction for L_pair; independent noise, K=4 samples for L_dist. L_pair is mean valid-pair RMS Euclidean in 25 raw output dimensions. Total loss is L_pair + 1.0*L_dist. No sample/no_grad wrapper is used for training.', '',
'B2 matches each generated descriptor distribution to its paired singleton. B3 matches each input separately to the same-session unique reference set. B4 cross averages banks (occurrences 0/1 and 2/3) against the references, while its self is ONLY the full cross-bank mean. It does not use an iid flattened self statistic. Source repetitions across banks are permitted because occurrences are drawn independently with replacement.', '',
'## Descriptor and scaler', '',
'phi has 75 coordinates: temporal mean(25), population temporal std(25), first-difference RMS(25). Valid frames only; length-one velocity has an explicit false coordinate mask. Distances use only jointly valid coordinates. Vector norms use finite zero subgradients, and no epsilon pseudo-velocity is inserted. Descriptors do not learn parameters.', '',
'Prediction descriptors use source_lengths in all three arms. Paired target descriptors use pair_lengths; the central correspondence mask also uses pair_lengths. Independent listener references use their own center crops and lengths. No framewise alignment is assumed between independent interactions.', '',
'Frozen train-only scaler: (phi-mean_train)/max(std_train,1e-4), fit to all 1660 unique TRAIN listener clips with independent T=128 center crops. Invalid coordinates are excluded from fitted moments/counts; scaler tensors are buffers, not trainable parameters. VAL never refits. Exact source file SHA-256 and crop metadata are included in descriptor_stats.json.', '',
'File SHA-256: `'+manifest['scaler_hash']+'`.', '',
'Train scaler source-list hash: `'+manifest['train_scaler_source_hash']+'`.', '',
'## Sampling and exact controls', '',
'After split filtering, the train pool has 17 sessions, 1660 unique speaker sources and 1660 unique listener refs. Each step chooses session with p(s)=n_s/1660, then four IID source occurrences with replacement. References are up to eight unique clips uniformly without replacement, including paired listeners when selected. Reference crops are computed independently from source offsets. No ref slots are duplicated.', '',
'128 steps produce 512 source occurrences / '+str(sampling['unique_sources'])+' unique sources. '+str(sampling['steps_with_repeated_source'])+' steps contain repeated source IDs, as allowed. B3/B4 each use 1024 reference occurrences / '+str(sampling['unique_sampled_references'])+' unique refs. B2 training performs zero population reference lookups; shared train-only scaler fitting is the explicit common preprocessing.', '',
table(['Schedule component','SHA-256'],list(schedule['hashes'].items())), '',
'Initialization hash: `'+trained['arms']['B2']['initialization_hash']+'`. Runtime consumed schedules/noise/scaler hashes, parameter counts, step counts and optimizer settings match across all arms; B3/B4 consumed reference hashes match the planned reference hash. Complete probabilities and empirical session counts are in sampling_audit.json.', '',
'Config: T=128, K_train=4, B=4, R<=8, steps=128, lambda_dist=1, AdamW lr=3e-4 / weight_decay=.01. Initialization seed=123, session=2001, source=2002, reference=2003, noise=789. GPU 2. No subsequent tuning, extra steps, best-checkpoint selection, or additional loss.', '',
'## Gradient preflight', '',
table(['Arm','Pair magnitude','Dist magnitude','Pair grad L2','Dist grad L2','Larger/smaller'],[[arm,f(a['pair_loss']),f(a['dist_loss']),f(a['pair_gradient_norms']['total']),f(a['dist_gradient_norms']['total']),f(a['gradient_ratio'])] for arm,a in preflight.items()]), '',
'Both terms were backwarded separately before optimization on the first matched batch. All ratios are below 100, so the stop gate passed without changing lambda. Module-specific norms are saved. During training, every total gradient is finite and every prior gradient is zero. Separate component gradient ratios were not remeasured on every later step.', '',
'## Fixed validation and estimator', '',
'Exactly the Phase 1.5 frozen 80 clip IDs (4 x 20 sessions), hash `'+manifest['validation_ids_hash']+'`. K=32 uses the saved Phase 1.5 noise-bank prefix, identical across inputs/arms. All 571 unique VAL listener references are used as Q_s, uniformly within session, with independent center crops. Results macro-average the 20 sessions.', '',
'Session marginal self correctly handles shared random numbers: equal epsilon indices are excluded for ALL input pairs. The estimator is the uniform mixture over four fixed sources, with independent epsilon draws; an ordinary iid self-term on all flattened correlated outputs would be biased. Tests cover this distinction. Evaluation target/ref information never enters the generator.', '',
'Conditionality gap is reported both as the requested ratio shuffled/correct and as shuffled-correct difference. A fixed same-session derangement (seed 1502) is saved. These are relative-input tests, not semantic claims about latent coordinates.', '',
'## Paired correspondence and population fit', '',
table(['Arm','Central paired RMS','AU paired','VA paired','Expression paired','Session ES','Per-input population ES'],[[arm,*[f(aggregate[arm]['pair'][key]) for key in ('overall','AU','VA','expression')],f(aggregate[arm]['session_marginal']['loss']),f(aggregate[arm]['per_input_population']['loss'])] for arm in arms]), '',
table(['Arm','Marginal cross','Marginal self','Marginal ES','Within-input descriptor self','Between-input descriptor distance'],[[arm,*[f(aggregate[arm]['session_marginal'][key]) for key in ('cross','self','loss')],f(decomposition[arm]['within_input_descriptor_self']),f(decomposition[arm]['between_input_descriptor_distance'])] for arm in arms]), '',
'For four equally weighted source components, marginal self = 1/4 within-input self + 3/4 between-input distance (all under distinct noise indices). This decomposition is algebraic from the same estimator; no new model run was performed.', '',
'## Input-shuffle conditionality', '',
table(['Arm','Score','Correct','Shuffled','Ratio','Difference'],[[arm,label,*[f(aggregate[arm][key][metric]) for metric in ('correct','shuffled','ratio','difference')]] for arm in arms for label,key in [('Central RMS','central_conditionality'),('Paired descriptor ES','paired_descriptor_conditionality')]]), '',
'Session bootstrap uses 10,000 resamples of 20 paired sessions, seed 2004. It quantifies variation across these selected sessions only, not training-seed or shuffle-permutation uncertainty. Negative ES/RMS differences favor B4; positive shuffle-gap differences indicate stronger measured input sensitivity.', '',
table(['B4 minus baseline contrast','Mean','95% session-bootstrap interval'],[[name,f(value['mean']),str([float(f(v)) for v in value['ci95']])] for name,value in comparisons.items()]), '',
'## Diversity shape (raw reaction units)', '',
table(['Arm','Group','Cross','Self','Prediction std','Boundary fraction','Velocity abs','Velocity RMS','Expression entropy'],[[arm,group,*[f(aggregate[arm]['diversity'][group][key]) for key in ('cross_distance','self_distance','prediction_std','boundary_fraction','velocity_abs_mean','velocity_rms')],f(aggregate[arm]['diversity'][group]['expression_entropy']) if group=='expression' else '-'] for arm in arms for group in ('overall','AU','VA','expression')]), '',
'Raw trajectory metrics use valid paired frames; descriptors are separately standardized and must not be numerically conflated with this table. Boundary tolerance is 0.01 from output-domain endpoints. Velocity uses adjacent frames, with no assumed fps. Entropy is in nats. A self distance near 0.0001 can be just the unchanged epsilon floor, not genuine stochastic variation.', '',
table(['Arm','Raw stochastic abs','tanh residual abs','tanh saturation'],[[arm,*[f(aggregate[arm]['residual'][key]) for key in ('stochastic_raw_abs_mean','stochastic_residual_abs_mean','tanh_saturation_fraction')]] for arm in arms]), '',
'## Real-reference description', '',
'571 unique val references yield 20,157 unordered same-session pairs. Session-macro real-real normalized descriptor distance is '+f(real_a['real_real_descriptor_session_macro_mean'])+'. Per-session distance distributions and crop lists are saved. These are independent interactions under different speaker inputs, not repeated conditional samples for one X. No generated-self-to-real-self penalty was used.', '',
table(['Real group','Boundary fraction','Velocity abs','Velocity RMS','Expression entropy'],[[group,*[f(real_a['trajectories'][group][key]) for key in ('boundary_fraction','velocity_abs_mean','velocity_rms')],f(real_a['trajectories'][group]['expression_entropy']) if group=='expression' else '-'] for group in ('AU','VA','expression')]), '',
'Compared with the Phase 1.5 selected-80 descriptive baseline, this baseline now uses the FULL val reference population and session-macro weighting; its numbers should not be mistaken for the identical prior subset.', '',
'## Toy hierarchy sanity', '',
'Analytic expected descriptor ES on support {-1,+1}: initial B3 ES='+f(toy['initial_B3_ES'])+', B4 marginal ES='+f(toy['initial_B4_ES'])+'. B3 optimizes probabilities to '+str(toy['final']['B3']['probabilities'])+'; B4 retains '+str(toy['final']['B4']['probabilities'])+'. Both marginals are 0.5. This verifies the loss connection, not a real-data performance result.', '',
'## Runtime, tests and artifacts', '',
table(['Arm','Training seconds','Training peak MiB','Evaluation seconds','Evaluation peak MiB'],[[arm,f(trained['arms'][arm]['training_seconds']),f(trained['arms'][arm]['peak_allocated_bytes']/2**20),f(evaluations[arm]['evaluation_seconds']),f(evaluations[arm]['peak_allocated_bytes']/2**20)] for arm in arms]), '',
'Times use synchronized CUDA timers around arm compute loops. Shared CPU data loading, scaler fitting, hashing, checkpoint IO and report generation are excluded; peak memory is PyTorch allocated memory, not total device reservation.', '',
'Full test output: pytest_results.txt. Tests include split isolation and escape rejection, unique refs, independent crops, descriptor sample-order equivariance/masking/single-frame validity, train-only frozen scaler, hand-calculated B2 ES, B3 permutation invariance, B4 bank exchange and non-iid self, IID replacement schedules/independent noise, generator isolation, real consumed hashes, toy sanity and the original Phase 1/1.5/K-prefix suites.', '',
'Artifacts: descriptor_stats.json, training_manifest.json, schedule.json, training_noise.npy, sampling_audit.json, gradient_preflight.json, B2/B3/B4_training.jsonl, training_summary.json, toy_hierarchy.json, runtime_source.tar.gz, pytest_results.txt, evaluation/{B2,B3,B4}.json, summary.json, real_reference.json, manifest.json, common_noise_K32.npy, session_bootstrap.json, spread_decomposition.json, phase2_curves.png/svg, this report and artifact_inventory.json. Checkpoints are retained locally in checkpoints/{B2,B3,B4}.pt, Git-ignored; their hashes are in training_summary.json and the inventory. Inventory excludes itself to avoid a circular hash.', '',
'![Phase 2 curves](evaluation/phase2_curves.png)', '',
'## Unverified / not established', '',
'- No SOTA, official-test or generalization claim. One seed, 128 steps, fixed center crops and one shuffle permutation only.',
'- No evidence that B4 recovered a healthy stochastic conditional distribution: its within-input noise usage nearly disappears.',
'- Lower marginal ES does not establish correct distribution shape; B4 cross distance is worse and VA remains almost fully saturated.',
'- Larger central shuffle gap is not a proof of fine correspondence; absolute AU/expression pairing worsens and descriptor shuffle effects are small.',
'- The hierarchy change causes the observed differences under matched controls, but improvement independent of increased between-input spread is not established.',
'- No per-step re-audit of separate pair/dist gradient ratios after the initial gate; only finite total-gradient monitoring.',
'- K64, multiple seeds, long training and official test were not run. No prohibited module, auxiliary loss, lambda sweep or structural adjustment was added.', '',
'Replay evaluation into a new directory:', '',
'```bash',
'.venv/bin/python -m hirp.evaluate_phase2 --run runs/phase2 --output runs/phase2/evaluation_repeat --device cuda:2',
'```', '']
    (run/'PHASE2_REPORT.md').write_text('\n'.join(lines))
    inventory=[]
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name!='artifact_inventory.json':inventory.append(dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=sha256_file(path)))
    (run/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2))
    print('COMPARISONS',json.dumps(comparisons),flush=True)
    print('REPORT',run/'PHASE2_REPORT.md','ARTIFACTS',len(inventory),flush=True)


if __name__=='__main__':main()
