# HiRP Phase 2.1: conditional distribution supervision

Base / local HEAD at resumption: `8fbc5febddb14b0b54d27633a3edb95227cc9834`; branch `agent/hirp-phase21-conditional-score`. No later commits existed. Existing uncommitted work was retained. See resumption_audit.json and code_changes.patch. No push.

## Findings

The pre-norm output path restores measurable stochastic trainability in this bounded experiment. C0 has the best held-out paired trajectory ES. C2 improves group descriptor ES relative to C0 while worsening conditional ES; compared with C1, C2 improves both conditional and marginal ES. All three correct-input scores beat the fixed within-session shuffle. This supports input correspondence sensitivity, not recovery of each full conditional distribution.

Self distance is a legitimate term of ES. A self-driven improvement is not by itself gaming; neither is improved marginal ES proof of conditional calibration. AU/expression spread remains small and expression entropy remains high relative to real references. No complete mechanism or MARS performance claim is established.

## Objectives and controls

| Experiment | Paired term | Distribution term |
|---|---|---|
| Historical B2/B3/B4 | RMS(G(X,0),Y) | paired descriptor / per-input population / cross-bank group ES |
| C0 | trajectory ES over every stochastic sample | none |
| C1 | same trajectory ES | 0.1 × unchanged b3_score |
| C2 | same trajectory ES | 0.1 × unchanged b4_score |

G(X,0) is diagnostic only, neither an expectation nor a median. Trajectory scores use pair_lengths and 25 unit channel scales; generated descriptors use source_lengths and the frozen train-only 75-coordinate scaler. Targets/references do not enter forward. A0 prior is frozen; encoder/decoder remain conditional on X.

All arms: scratch seed 123, T=128, K_train=4, B=4, 128 steps, AdamW lr=1e-4, weight decay .01, float32, pre-norm with default projection initialization, 8,729,496 parameters. FiLM small nonzero initialization and output domains/residual formula are preserved. The optional small projection initialization was implemented/tested but not used for the pilot.

The original Phase 2 schedule and independent noise were hash-verified and reused. Four source occurrences are IID with replacement, split into two banks; references are unique within a step and train-only. C1/C2 consumed identical references. C0 has zero reference lookups in its loss/step path; the shared experiment preparation loads a reference cache for C1/C2. This is not three separately isolated data-loading processes.

Lambda was selected from [.01,.03,.1,.3,1] on 8 TRAIN batches × 2 arms, nearest log-distance to a median weighted group/conditional gradient ratio of .5. Selected .1; actual initial median ratio .771551. This is a reasonable fixed starting point, not optimality or balance evidence. Gradient norms/cosines recur every 16 steps; no dynamic controller.

## Output-path evidence

32-step old/pre-norm smoke uses the same conditional objective, projection/backbone initialization and learning rate. The locked TRAIN-only viability rule rejected old-head and accepted pre-norm. Historical B2/B3/B4 checkpoints were only probed read-only; no structural checkpoint migration.

| Probe at final step | Residual saturation | Median tanh derivative | AU noise VJP | VA noise VJP | Expression noise VJP |
|---|---:|---:|---:|---:|---:|
| old_head | 0.7775195 | 0.0005606413 | 0.0002989457 | 3.524158e-08 | 0.001350199 |
| pre_norm_head | 0.0002148437 | 0.3142027 | 0.0004505719 | 0.008512069 | 0.0004437803 |
| C0 | 0.002753906 | 0.2740361 | 0.004041892 | 0.2409069 | 0.004444585 |
| C1 | 0.001210938 | 0.2463742 | 0.00982562 | 0.08030634 | 0.01503286 |
| C2 | 0.0001757812 | 0.3690478 | 0.006683987 | 0.2346664 | 0.005591424 |

Full *_path_diagnostics.json records encoder H, decoder hidden, base/stochastic logits and final raw: valid-frame mean/std/absolute quantiles and conditional gradient norms, plus boundaries, sample std and derivative quantiles. Noise values are seeded normalized random-output VJPs, not full Jacobian norms. Prior parameters remained unchanged. These fixed TRAIN probes are not whole-dataset saturation estimates.

| Historical arm | Base VA abs p95 | Final VA abs p95 | Residual saturation |
|---|---:|---:|---:|
| B2 | 11.8007 | 12.205 | 0.974609 |
| B3 | 15.9715 | 16.975 | 0.855234 |
| B4 | 21.3408 | 22.3426 | 0.859414 |

Base and stochastic paths must both be considered: bounded stochastic residuals cannot undo arbitrarily large base logits. Pre-norm normalizes both projections; the evidence does not isolate which normalization is individually necessary.

| Arm | Weighted group/conditional gradient ratio min–max | Gradient cosine min–max |
|---|---:|---:|
| C1 | 0.133692–0.620465 | -0.417233–0.866566 |
| C2 | 0.211074–1.01088 | -0.168347–0.942982 |

## Held-out fixed-80 pilot

K=32 common Gaussian bank; 4 clips × 20 sessions, all 571 unique VAL listener references. Session-macro averages; common-mask correct/shuffled comparisons. Marginal self excludes equal noise indices for every source pair. No official metrics or hidden test.

| Arm | Paired ES | Cross | Self | Shuffled ES | Shuffle gap | Group ES | Central RMS (aux) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.2801323 | 0.3396242 | 0.1189838 | 0.2953634 | 0.0152311 | 0.9739674 | 0.3254306 |
| C1 | 0.3092583 | 0.3276570 | 0.0367976 | 0.3220405 | 0.0127823 | 1.0137597 | 0.3263801 |
| C2 | 0.2931024 | 0.3345144 | 0.0828242 | 0.3141267 | 0.0210243 | 0.9419656 | 0.3314854 |

| Arm | Marginal cross | Marginal self | Within-input descriptor self | Between-input descriptor distance |
|---|---:|---:|---:|---:|
| C0 | 1.1602310 | 0.3725273 | 0.3103265 | 0.3932609 |
| C1 | 1.1226070 | 0.2176947 | 0.1438818 | 0.2422990 |
| C2 | 1.1704248 | 0.4569183 | 0.3409949 | 0.4955594 |

For four fixed sources, marginal self = within/4 + 3×between/4, excluding shared noise indices. Descriptor units differ from raw trajectory units.

| Contrast | Mean | Session bootstrap 95% interval |
|---|---:|---|
| C0_shuffle_gap | 0.0152311 | [0.0101529, 0.0206330] |
| C1_shuffle_gap | 0.0127823 | [0.0080969, 0.0177542] |
| C2_shuffle_gap | 0.0210243 | [0.0140532, 0.0284564] |
| C2_minus_C0_conditional | 0.0129701 | [0.0108281, 0.0155723] |
| C2_minus_C0_marginal | -0.0320018 | [-0.0444740, -0.0176659] |
| C2_minus_C1_conditional | -0.0161559 | [-0.0202572, -0.0120302] |
| C2_minus_C1_marginal | -0.0717941 | [-0.0814768, -0.0609054] |

Bootstrap uses 10,000 paired session resamples, seed 2106. It excludes training-seed, noise-bank and shuffle-permutation uncertainty.

| Arm/channel | Sample std | Boundary fraction | Velocity RMS | Expression entropy |
|---|---:|---:|---:|---:|
| C0/AU | 0.002792103 | 0 | 0.01546588 | — |
| C0/VA | 0.2633947 | 0 | 0.04652022 | — |
| C0/expression | 0.003850807 | 0 | 0.01218489 | 1.8151414081454278 |
| C1/AU | 0.004253377 | 0 | 0.02317434 | — |
| C1/VA | 0.07480335 | 0 | 0.03426398 | — |
| C1/expression | 0.009963354 | 0.004283905 | 0.02322202 | 1.8256791919469832 |
| C2/AU | 0.003807775 | 0 | 0.02754631 | — |
| C2/VA | 0.179214 | 0 | 0.04350344 | — |
| C2/expression | 0.004813771 | 0.009358978 | 0.02407513 | 1.7669732138514518 |

Real reference distributions/crops are saved in evaluation/real_reference.json; per-input and session metrics plus held-out path probes in evaluation/C*.json. References describe different X, not repeated conditional draws at one X.

## Toy recovery

Same-session Bernoulli probabilities [.1,.9], 8192 sampled labels per input, seed 2103. A two-input linear sigmoid network optimizes exact expected ES or the full sampled-label empirical objective. Initializations [.5,.5], [.2,.4], [.8,.3]; 500 steps. This validates the objective connection, not REACT generalization.

| Initialization | Objective | Arm | Recovered probabilities | Conditional MAE | Marginal error |
|---|---|---|---|---:|---:|
| [0.5, 0.5] | analytic_expected | C0 | [0.1, 0.9] | 6.312061e-09 | 0 |
| [0.5, 0.5] | analytic_expected | C1 | [0.1363636, 0.8636364] | 0.03636364 | 0 |
| [0.5, 0.5] | analytic_expected | C2 | [0.1, 0.9] | 6.312061e-09 | 0 |
| [0.5, 0.5] | sampled_labels_logistic_network | C0 | [0.1030273, 0.9016113] | 0.002319322 | 0.002319322 |
| [0.5, 0.5] | sampled_labels_logistic_network | C1 | [0.1391158, 0.8651012] | 0.03700728 | 0.002108487 |
| [0.5, 0.5] | sampled_labels_logistic_network | C2 | [0.1028165, 0.9014005] | 0.002108484 | 0.002108484 |
| [0.2, 0.4] | analytic_expected | C0 | [0.1, 0.8999936] | 3.193705e-06 | 3.193705e-06 |
| [0.2, 0.4] | analytic_expected | C1 | [0.1363636, 0.8636364] | 0.03636364 | 3.20502e-11 |
| [0.2, 0.4] | analytic_expected | C2 | [0.1000003, 0.8999938] | 3.23339e-06 | 2.924497e-06 |
| [0.2, 0.4] | sampled_labels_logistic_network | C0 | [0.1030273, 0.9016022] | 0.002314773 | 0.002314773 |
| [0.2, 0.4] | sampled_labels_logistic_network | C1 | [0.1391158, 0.8651012] | 0.03700728 | 0.002108487 |
| [0.2, 0.4] | sampled_labels_logistic_network | C2 | [0.1028169, 0.901392] | 0.002104474 | 0.002104474 |
| [0.8, 0.3] | analytic_expected | C0 | [0.1002384, 0.8999297] | 0.0001543267 | 8.405016e-05 |
| [0.8, 0.3] | analytic_expected | C1 | [0.1363643, 0.8636364] | 0.03636398 | 3.453282e-07 |
| [0.8, 0.3] | analytic_expected | C2 | [0.1002249, 0.8999134] | 0.0001557725 | 6.913625e-05 |
| [0.8, 0.3] | sampled_labels_logistic_network | C0 | [0.1031983, 0.9015233] | 0.002360813 | 0.002360813 |
| [0.8, 0.3] | sampled_labels_logistic_network | C1 | [0.1391161, 0.8651012] | 0.03700745 | 0.00210865 |
| [0.8, 0.3] | sampled_labels_logistic_network | C2 | [0.1029863, 0.9013076] | 0.002146909 | 0.002146909 |

Analytic C0/C2 optimum is [.1,.9]; C1 optimum is [.1363636,.8636364] at lambda=.1. The original optimum-initialized compatibility toy is unchanged.

## Validation, provenance and remaining work

Actual earlier stage commands and failures: commands.jsonl. Earlier full test result: 101 passed, 1 xfailed, 1 warning in pytest_before_pilot.txt. Resumed sandbox tests: 98 passed, 3 CUDA skipped, 1 xfailed, 1 warning. GPU-enabled repeat: 101 passed, 1 xfailed, 1 warning in 13.74s (pytest_resumed_gpu.txt). CPU BF16 fused Transformer eval/no_grad on PyTorch 2.1 is explicitly xfailed; pilot/evaluation use float32, and ES distances remain float32 under AMP.

Resumed evaluation command: `.venv/bin/python -m hirp.evaluate_phase21 --run runs/phase21/conditional_score_v0 --device cuda:2`. Full GPU test command: `.venv/bin/python -m pytest hirp/tests -q`. Report/bootstrap command: `.venv/bin/python -m hirp.summarize_phase21`.

83 historical artifact hashes match the preparation snapshot. Runtime smoke/training archives and checkpoint hashes are retained. No old code/report/data/checkpoint/official metric changes. Development failure logs are preserved. Checkpoints are local Git-ignored files.

Not run: long training, multi-seed pilot, K=64 sensitivity, multiple shuffles/noise banks, hidden/official evaluation, small-projection-init training comparison, separate base-only/stochastic-only normalization ablations, conditional-prior learning, or push. Neither learning rate nor lambda is claimed optimal. The existing bounded pilot was reused, not repeated after validation.
