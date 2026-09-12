# T0 next decision — completed focused analysis

**Retain native T0 as the provisional Flow parent. Next, test one explicitly paired-time conditioning intervention, not more spread pressure.** T0 exceeds the archived MAM FRC80 and native exact FRD20 reference, but its S-MSE is about half of MAM. Fixed AU thresholding closes only about 11.4% of this absolute S-MSE gap. The missing spread is predominantly candidate temporal means: T0 already has slightly more centered P8 slow spread than MAM. Four final-T0 source probes show mixed task-relevant conditioning, not a stable paired advantage. This is a small diagnostic, not proof of independence.

## Completed native systems (unchanged policies)

| Model | FRC80 ↑ | native exact FRD20 ↓ | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| MAM | 0.810962319 | 172.576423473 | 0.157203704 | 0.058911592 |
| R2 | 1.181468685 | 133.409074259 | 0.043679938 | 0.039064668 |
| A12000 | 0.414712854 | 154.005436547 | 0.139438540 | 0.068698995 |
| T0 | 0.859622576 | 133.876667407 | 0.081215642 | 0.053756177 |
| T1 | 0.833056843 | 133.433357505 | 0.082931809 | 0.052843958 |

Every row uses its own single checkpoint and native policy. MAM has native rounded AU and a different training budget; R2/Flow use continuous AU. T0/T1 share A12000 plus2000 updates, inheriting R2-6000 encoder cost; no MAM weights are used. T1 remains a small distance/spread tradeoff, not the preferred parent. Completed native metrics and FRD pairs were read, not rerun.

## Where candidate spread differs

Equal weight per source (80), K10, full valid recording. G is each candidate's temporal mean; P8 uses true tail lengths. DC=G, slow=P8−G, fast=Y−P8. All components use the same K(K−1)TD denominator. These names denote time scales, not artifacts, semantic modes or quality.

| Model | algebraic total | DC | centered slow | fast |
|---|---:|---:|---:|---:|
| MAM | 0.157206 | 0.090987 | 0.050809 | 0.015410 |
| R2 | 0.043682 | 0.015789 | 0.025819 | 0.002074 |
| A12000 | 0.139440 | 0.010605 | 0.081507 | 0.047328 |
| T0 | 0.081217 | 0.008237 | 0.055948 | 0.017032 |
| T1 | 0.082933 | 0.008202 | 0.057046 | 0.017684 |

Max per-source/group reconstruction error: **1.39e-16**. Group-weight reconstruction also passed <1e-12. Double-precision algebraic totals differ from archived FP32 torch.cdist S-MSE by roughly1–2e-6; native metrics above are unchanged.

Grouped cells below give **within-group normalized value / contribution to all25 channels**. Contribution weights are AU15/25, VA2/25, expression8/25; contributions sum to the overall row above.

| Model/group | total | DC | slow | fast |
|---|---:|---:|---:|---:|
| MAM/AU | 0.18161 / 0.10897 | 0.08477 / 0.05086 | 0.07252 / 0.04351 | 0.02432 / 0.01459 |
| MAM/VA | 0.11867 / 0.00949 | 0.11193 / 0.00895 | 0.00620 / 0.00050 | 0.00053 / 0.00004 |
| MAM/expression | 0.12109 / 0.03875 | 0.09740 / 0.03117 | 0.02125 / 0.00680 | 0.00243 / 0.00078 |
| R2/AU | 0.06007 / 0.03604 | 0.02080 / 0.01248 | 0.03636 / 0.02182 | 0.00290 / 0.00174 |
| R2/VA | 0.00343 / 0.00027 | 0.00108 / 0.00009 | 0.00212 / 0.00017 | 0.00023 / 0.00002 |
| R2/expression | 0.02303 / 0.00737 | 0.01007 / 0.00322 | 0.01197 / 0.00383 | 0.00098 / 0.00031 |
| A12000/AU | 0.17227 / 0.10336 | 0.01182 / 0.00709 | 0.09554 / 0.05733 | 0.06490 / 0.03894 |
| A12000/VA | 0.05749 / 0.00460 | 0.00704 / 0.00056 | 0.04086 / 0.00327 | 0.00958 / 0.00077 |
| A12000/expression | 0.09837 / 0.03148 | 0.00921 / 0.00295 | 0.06535 / 0.02091 | 0.02381 / 0.00762 |
| T0/AU | 0.09630 / 0.05778 | 0.00932 / 0.00559 | 0.06439 / 0.03863 | 0.02260 / 0.01356 |
| T0/VA | 0.02978 / 0.00238 | 0.00289 / 0.00023 | 0.02305 / 0.00184 | 0.00383 / 0.00031 |
| T0/expression | 0.06580 / 0.02106 | 0.00755 / 0.00242 | 0.04835 / 0.01547 | 0.00990 / 0.00317 |
| T1/AU | 0.09927 / 0.05956 | 0.00928 / 0.00557 | 0.06545 / 0.03927 | 0.02453 / 0.01472 |
| T1/VA | 0.02949 / 0.00236 | 0.00297 / 0.00024 | 0.02354 / 0.00188 | 0.00298 / 0.00024 |
| T1/expression | 0.06567 / 0.02101 | 0.00748 / 0.00239 | 0.04966 / 0.01589 | 0.00853 / 0.00273 |

T0 vs MAM: DC0.00824 vs0.09099, slow0.05595 vs0.05081, fast0.01703 vs0.01541. The DC deficit is larger than the net total deficit because the other components partly offset it. In AU alone, slow is lower for T0; in VA/expression it is higher. This does not prescribe maximizing DC or require MAM's candidate biases to be copied.

## Target matching and candidate quality

Native saved10×10 CCC matrices, preserving all target slots and weights. Best-CCC distribution pools800 candidate values for description, not800 independent subjects. Target-side best match is not a mode count.

| Model | target-side mean best CCC | candidate best CCC p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| MAM | 0.042801 | 0.025374 | 0.072293 | 0.149866 |
| R2 | 0.061171 | 0.044873 | 0.111009 | 0.195037 |
| A12000 | 0.037716 | 0.013657 | 0.036440 | 0.076396 |
| T0 | 0.058758 | 0.031626 | 0.074823 | 0.157531 |
| T1 | 0.059138 | 0.031958 | 0.072650 | 0.152668 |

T0 has better target-side matching than MAM (0.05876 vs0.04280) despite less raw spread. This is meaningful target-pool coverage evidence, not recovery of conditional probabilities. Duplicate references remain slots, not independent modes.

Speed / acceleration (mean absolute first / second frame differences), separate populations:

| Population | AU | VA | expression |
|---|---:|---:|---:|
| MAM | 0.02040 / 0.04079 | 0.02891 / 0.04243 | 0.01475 / 0.02243 |
| R2 | 0.01529 / 0.02178 | 0.02296 / 0.02869 | 0.01248 / 0.01732 |
| A12000 | 0.08023 / 0.15575 | 0.06888 / 0.11873 | 0.05623 / 0.10088 |
| T0 | 0.03409 / 0.06402 | 0.04208 / 0.07229 | 0.03845 / 0.06722 |
| T1 | 0.03614 / 0.06806 | 0.03589 / 0.06150 | 0.03402 / 0.05935 |
| processed_targets | 0.01088 / 0.02176 | 0.01547 / 0.01949 | 0.01187 / 0.01617 |
| raw TRAIN16 paired crops | 0.02062 / 0.04122 | 0.03606 / 0.05513 | 0.02493 / 0.04066 |

Processed Development targets and raw TRAIN crops differ in population, lengths and processing. Neither is a per-input variance target or a universal human dynamic ceiling. T0 still moves faster than processed targets in all groups; its dynamics are not uniformly matched simply because FRC/FRD improved.

## One AU-policy diagnostic (not a new production policy)

Exactly AU>=0.5 for **every listed model**, VA/expression untouched, all10 candidates, original targets and official FRC/S-MSE/FRVar implementations. No new generation or policy search.

| Model | diagnostic FRC80 | diagnostic S-MSE80 | diagnostic FRVar80 | diagnostic exact FRD20 |
|---|---:|---:|---:|---|
| MAM | 0.810962319 | 0.157203704 | 0.058911592 | not run | 
| R2 | 1.091605410 | 0.075637750 | 0.057272222 | not run | 
| A12000 | 0.409351888 | 0.146943092 | 0.072442636 | not run | 
| T0 | 0.842509286 | 0.089873716 | 0.058232527 | not run | 
| T1 | 0.816606819 | 0.091199622 | 0.057093177 | not run | 

T0's thresholded S-MSE gain0.00866 is only11.4% of its native0.07599 gap to MAM; T1 closes about11.1%. Both FRCs decrease. Thus output convention is a material but minor contributor, not the main explanation. No native FRD is transferred into this table. MAM happens to be unchanged by this policy; its native FRD remains exclusively in the native table.

## Final T0 conditional probe (not A's evidence)

Recipients0/20/40/60; same cyclic within-session donors1/21/41/61. Cached correct exports were verified against T0 checkpoint, manifest, noise-content hash, FP64 precision, Euler16 and750-block protocol; only four donor forwards were generated. Recipient absolute-frame noise is unchanged, including common prefixes when lengths differ. Common masks truncate scores only; generator sees donor source lengths, no target information.

| Recipient/donor | common frames | paired CCC correct/shuffled | multireference FRC correct/shuffled | mean output change |
|---|---:|---:|---:|---:|
| 0/1 | 1167 | 0.00455 / 0.03401 | 0.38241 / 0.63183 | 0.06938 |
| 20/21 | 1020 | -0.01594 / -0.03995 | 1.27725 / 1.16113 | 0.07142 |
| 40/41 | 271 | 0.02488 / 0.03334 | 0.66043 / 0.63243 | 0.06406 |
| 60/61 | 1892 | 0.01166 / 0.01279 | 0.71182 / 0.62037 | 0.05034 |

Correct paired CCC wins1/4; multireference FRC wins3/4. Mean paired CCC difference is negative on this tiny fixed sample. Output changes alone do not establish useful conditioning; these four cases do not establish dataset-wide failure or statistical significance either.

## One next controlled hypothesis — proposal only

**Test a small nonzero same-frame speaker projection W_h H_t into the velocity token input, retaining cross-attention and T0's task/FM objective.** Motivation: task quality exists and slow centered spread is not globally deficient, while paired source correspondence is not consistently demonstrated. A paired-time path may make local speaker information easier to use than cross-attention alone. Compare with an equal-budget unchanged-T0 continuation from the same parent, preserving data/noise/optimizer/output/solver; judge paired response plus native FRC/FRD and decomposed diversity jointly. Do not simultaneously alter noise, add DC/dispersion rewards, change AU policy or increase T1's dynamic weight. This is a documented hypothesis, **not implemented or launched** in this batch and not guaranteed to work.

## Reproducibility and scope

- Exact result/export/target/metric-source hashes: `references.json`; T0 probe checkpoint and source IDs: `T0_condition_probe.json`; T1 checkpoint hash verified in `execution.json`.
- T0 checkpoint: `e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6`.
- T1 checkpoint: `69984e99746f2fc6b77a5de246c92f3236c5a98d6b1d7ff7fab5e67d5806a6f3`.
- Raw small CSVs: `decomposition_per_source.csv`, `decomposition_summary.csv`, `candidate_best_CCC.csv`, `matching_summary.csv`, `target_matching.csv`, `AU_policy_diagnostic.csv`, `native_scores.csv`, `raw_TRAIN_activity_summary.csv`, `T0_condition_probe.csv`.
- Commands and code hashes: `execution.json`. Initial CPU metric run hit sandbox multiprocessing permission; unchanged script completed on host. Original constant-channel warnings from official CCC were retained; finite final outputs verified.
- No new training/checkpoint, native-metric rerun, FRD rerun, additional noise bank/seed, hidden-test evaluation, deletion or push. New artifacts live only in `runs/reaction_flow/t0_next_v1` with two new analysis scripts.
- Seed123, repeatedly used Development80 and fixedFRD20 subset, not independent confirmation or equal-compute MAM comparison.
