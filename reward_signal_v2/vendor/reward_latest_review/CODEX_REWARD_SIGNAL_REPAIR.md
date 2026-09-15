# CODEX: repair reward signal before changing the model

## Scope and status

Reviewed repository: `william1111111111/React2025-new`.
Reviewed HEAD: `e7cb665c80933a6ca9e3a2d4bb19a0693b5e4668` on `agent/bert-semantic-controls`.
Read the actual local HEAD first. Reuse newer completed work; do not overwrite it with this snapshot.

This task is NOT a new architecture proposal and NOT permission to expand GPU/paid-API use.
Preserve the frozen P2 generator, candidate count K=10, official metrics, target manifests,
and existing TRAIN/DEV separation. Do not change running M0/M1/BERT experiments.
Use a new `reward_signal_v2` artifact directory and explicitly version any new reward.
No automatic push. No multi-seed or hyperparameter grid.

## Findings to preserve accurately

1. The generator is frozen by design. The optimizer contains the sampling policy only.
2. `policy.log_prob(actions.detach(), context, mask)` is a valid score-function path.
   Do not remove detach or differentiate through hard DTW to “fix” REINFORCE.
3. The three policies each completed 500 updates / 10,000 action rollouts, but output
   metrics are nearly unchanged. This does not mean gradients or optimizer steps were zero.
4. On the three saved R-coverage checkpoint500 TRAIN diagnostic batches, coverage-only
   gradient norms are about 6.48e-6, 4.05e-90 and .04147, versus quality norms 16.66,
   3.23 and 24.28. These are fixed-checkpoint diagnostics, not a complete training history.
5. Weighted KL gradients are about 0.25%–1.52% of total actor gradient on those batches.
   Do not conclude that KL is the main obstacle or remove it without an intervention.
6. `pilot.py` calibrates feature scales from generated reference phi only. Some scales
   are .001. `rewards.py` uses exp(-mean(zscore**2)/2) with no distance-bandwidth calibration.
   The saved reachability gate tests utility > 0: 8 of the 11 passing cases have coverage
   no larger than about 4.9e-21. This is not a meaningful reward-signal acceptance check.
7. Leave-one-out quality terms still contain absolute c/sc and d/sd offsets. The estimator
   is not invalid just because of this; it can have avoidable score-function variance.
8. R-distance has a material bonus gradient in its saved diagnostic. Coverage underflow
   cannot alone explain all arms. Policy displacement and action-to-output sensitivity
   still need to be measured.

## Phase A: reuse the saved data; no generator training

Read saved calibration (64 episodes), reachability (16 cases), reference tables and
TRAIN reward logs. Do not rerun existing official FRD or downloaded teachers.
Report, separately before and after gates:
- CCC threshold pass rate; DTW pass rate; joint SAME-target pass rate;
- domain/speed/acceleration pass rate (not interchangeable with pair quality gates);
- normalized squared-distance quantiles, coordinate/group contributions;
- exponent and similarity quantiles, exact zeros vs representable tiny positives;
- target columns with any eligible candidate;
- total coverage, per-candidate marginal gain, nontrivial-gain fraction;
- within-source candidate quality variance vs source-dependent offsets.
Keep masks for incomplete low-frequency bases. Do not average invalid padded modes.

Old c/d thresholds use reference best CCC and reference minimum DTW, which can choose
DIFFERENT targets. Requiring both on the SAME pair is a separate, stronger test;
measure this difference rather than assuming a gate rate.

For the initial and final policy, run a finite TRAIN-only fixed-input diagnostic:
- policy parameter hash / L2 change / relative change;
- same-base-noise action delta, mean/std and masked covariance;
- context-dependent shift/scale output and saturation;
- recomposed full-noise delta; delta on the actually rewarded block;
- prediction delta and reward-component delta;
- reward-only, quality-only, weighted-KL gradient norms and cosine;
- true optimizer delta before/after step, not just preclip gradient norm.
A no-update diagnostic must be labeled no-update. Finite-sample KL estimates may be
negative; do not present them as exact population KL or force-clamp the training estimator.

## Phase B: two isolated repairs

### B1. Quality baseline and numerically separated credits

Let per-candidate quality utility be
q[k] = dual_c * c[k]/s_c - dual_d * d[k]/s_d - dual_bad * bad[k].
Use
baseline[k] = sum(q[j] for j != k)/(K-1)
A_quality[k] = (q[k] - baseline[k])/K.

Conditional independence of candidates given source is required. The baseline must not
use candidate k's action, reward, or a joint-candidate generator dependency. An independent
frozen-reference quality mean is an alternative; select ONE beforehand, do not mix adaptively.
Use detached baseline and advantages. Preserve the fixed original K denominator.

Compute coverage gains separately:
A_cov[k] = coverage(U) - coverage(U without row k).
Then combine A[k] = A_cov[k] + A_quality[k]. Do not compute two large mixed sums then subtract;
that can numerically erase a tiny coverage term. Do not group-center the coverage marginal
gains as though they were independent per-candidate rewards.

This is a control variate, not an additional quality sacrifice or removal of constraints.
Keep the existing dual violation computation and c/d scales for the first comparison.

### B2. Calibrate useful similarities using TRAIN distances

Do not multiply an underflowed reward by an arbitrarily enormous coefficient.
First retain the original phi definition and quality gate for a one-variable bandwidth test:
D2[k,j] = mean(((phi[k]-target_phi[j])/s_phi)**2 over VALID coordinates).
On a predeclared calibration partition of saved TRAIN data, compute qualified-pair D2.
Choose one global h2 = median(D2_qualified)/(2*log(2)), with an explicit safe positive floor.
Then similarity = exp(-D2/(2*h2)). Median qualified-pair similarity is now near .5 on the
fit partition by construction; this is a scale convention, NOT calibrated correctness.
Freeze h2; test the remaining TRAIN calibration records without changing it.

Check whether one nearly constant coordinate dominates the distance and whether the
similarity has meaningful candidate-to-candidate differences. If bandwidth-only becomes
nearly constant or the feature geometry is dominated by degenerate channels, stop before
formal training. In a separately versioned metric, fit robust per-coordinate scales from
both generated-reference and real TRAIN target features, use group-aware floors/masks and
report all group contributions. No DEV-driven coordinate deletion or best-group selection.
Do not claim identical objectives when changing feature geometry.

Keep hard domain constraints. Do not immediately soften CCC/DTW gates at the same time.
If gates alone eliminate almost all pairs, bandwidth cannot help: report that finding,
then propose a separate soft quality margin experiment. Never convert every pair to eligible
merely to make rewards nonzero. Keep the old reporting quality criteria regardless.

The current speed/accel thresholds are almost the theoretical range maxima (~2 and ~4).
Report that bad_rate=0 is not evidence of realistic motion. Audit group-wise TRAIN real
motion statistics before interpreting any future spread gain; do not maximize these metrics.

## Phase C: finite on-policy comparison, not another open-ended search

Run only after B1/B2 produce finite, distinguishable rewards and valid score gradients.
All arms restart from the SAME identity sampling-policy initialization over the SAME frozen P2.
Do not compare a repaired arm with an old-policy endpoint that had a different initialization.

R0-baselined: old kernel + B1 quality baseline.
R1-calibrated: same B1 estimator + frozen TRAIN-calibrated kernel.

Reuse the same original first 100 update records (2 sources/update, K10), local/base-noise
keys, context, parent, optimizer type, LR1e-5, beta_KL=.01 and clipping norm1.
This isolates reward calibration while both arms use the corrected variance-reduction choice.
Exactly 100 updates each; do not extend until the observed signal chain is reported.
Do not claim per-update rollout costs exclude the existing/fresh independent reference draws;
account for both action and reference generation, plus replay diagnostics.

Required measurements at 0/25/50/100:
- policy change and fixed-input action/output change;
- eligible-pair rates, affinity quantiles, marginal-credit distribution;
- separate quality/bonus/KL gradients, clipping, optimizer delta;
- independent TRAIN-probe score change (not only the just-used training group).
Keep evaluation generation source-only and K10. DEV may be evaluated at the fixed100 endpoint
but must not choose the reward scales, gates or action perturbations. Native full-source FRC
and fixed exact FRD20 remain separate from block-level training rewards. Do not mix endpoints.

If the policy changes substantially and rewards have stable credit, but output barely moves,
then inspect action leverage, not another reward coefficient. A future source-only finite
difference probe may perturb one low-frequency coefficient with common random numbers;
no GT-selected action is ever used in final inference. Larger LR or richer action spaces
are subsequent single-variable experiments, not bundled into this repair.

## No premature algorithm change

Do not switch to PPO/GRPO, decoder RL, event classes, planners or shared noise to bypass
an inactive reward. R-distance already shows that a large gradient norm alone is not sufficient.
Do not call a large negative actor scalar “bad loss” or nonzero gradients “successful learning”.
Do not interpret gradient clipping multiplier as an identical multiplier of AdamW updates.
Do not normalize 1e-200 numerical noise to unit advantages; repair signal scale first.

## Deliverables

1. Existing-data reward audit, exact scope and missing cached data listed.
2. Minimal estimator/kernel patches plus tests.
3. Fixed 100-update two-arm table with signal-chain diagnostics.
4. One diagnosis: reward starvation repaired / gates still sparse / low credit SNR /
   policy-action leverage insufficient. Multiple mechanisms may coexist.
5. No claim of final diversity improvement until same-checkpoint quality and diversity
   jointly demonstrate it. Do not manufacture “performance preserved” from a nearly identity policy.

## Relevant reviewed source files

`reward_policy/rewards.py`, `pilot.py`, `policy.py`, `train.py`, `environment.py`;
`runs/reaction_flow/reward_policy_v1/scales.json`, `REACHABILITY.json`, `RESULTS.json`,
`KL_gradient_diagnostic_1789456439.json`, `TRAIN_ACCEPTANCE.json`, and R-coverage training.jsonl.

Bundled `verify_reward_signal.py` is a local synthetic / transcribed-policy check only.
It does NOT load a REACT checkpoint, regenerate facial trajectories, or validate actual
annotation correctness. Seven assertions groups passed in this review environment.
