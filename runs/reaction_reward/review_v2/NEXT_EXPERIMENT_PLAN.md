# Proposed A/B: conditional generated-candidate quality prediction

Status: proposal only. No training, bank regeneration, generator update or RL
is authorized or performed by this review. Do not resume old 4000-step RMs.

## Shared data and scope

Keep v1 split and RM_fit-only input normalization. Current split has 17/4/3
recording-date components; it is not participant-disjoint. Previously examined
RM_audit is now a development diagnostic, not a fresh independent confirmation.
No audit metric, candidate or source is used to choose new fit windows or thresholds.

Before training, expand REAL RM_fit evidence to every eligible one of the1100
fit recordings: at most two nonoverlapping native <=750-frame windows, selected
by fixed source-hash order from valid block starts. Selection uses source length
and native PTS only, not RM scores or audit labels. Apply frozen review_v2 PTS,
activity, delay and donor-fold rules; insufficient support remains UNKNOWN.
Reconstruct balanced reciprocal context comparisons; donor metadata never enters
forward. The actual expanded valid counts must be reported and frozen before
training, not assumed to equal2200 or called independent interactions.

Generated supervision retains the already fixed256 fit,64 cal,96 development
windows and P2/N0 fit/cal versus N1 development-only sources. This first A/B does
not expand generated-bank scope or infer new candidates from generator names.
Original arrays were intentionally evicted by storage cleanup. Before training,
regenerate exactly the existing seeded bank and verify hashes against saved
provenance; no new labels or DTW are needed if exact reconstruction succeeds.
Report any mismatch and stop; do not silently replace cached scores.
Maximum regeneration cost14848 candidate windows; new candidate IDs budget0.
This is compute cost even though it adds no independent observations.

## Architecture and objectives

Both arms start from identical fresh seed123 source/listener stems,2+2 temporal
Transformer layers and2 bidirectional cross-attention layers, width256,8heads,
FFN1024,dropout.1. No P2 task encoder weights. Original context/time heads remain.

A: repaired RM-Gen. Effective-only preferred ranking with original evidence
weights and family multipliers context1,time.5,weak.25,generated.5. Generated
ranking uses the existing context+time score as a baseline, not as a proven judge.

B: identical shared backbone and auxiliary ranking. Add separate quality_C and
quality_D heads on concatenated pooled source/listener fused representations
(512->256 GELU->1 per head). Predict best-target candidate CCC and min-target
native weighted DTW independently. Quality targets come only from saved fit
C/D matrices, including candidates belonging to UNKNOWN/conflicting preference
pairs; UNKNOWN still never receives a ranking gradient.

Use zC=(C-fit_mean_C)/fit_std_C. Use raw n-frame DTW only to form
uD=log1p(D/sqrt(n)); zD=(uD-fit_mean_uD)/fit_std_uD. Fit the moments with each
source equally weighted and each date component equally weighted, not pairwise
presentation counts; floor std at1e-6. Preserve raw C,D,n and transform hash.
This stabilizes a training target; it does not redefine DTW or make it full FRD.

B objective = A auxiliary ranking + Huber(pred_zC,zC) + Huber(pred_zD,zD), each
head coefficient1, delta1, fixed before training; no coefficient grid. Do not
allow improvements in one target to erase reporting of errors in the other.
For candidate ordering, predefine predicted zC - predicted zD as a scalar
quality tradeoff, report both heads individually and Pareto consistency. Context
and time are auxiliary in B, not silently added into its quality selection score.
A uses context+time. Neither score is a human-correctness probability.

## Budget, sampling and selection

4000 maximum updates,seed123,AdamW1e-4,weight_decay.01,warmup200,clip1.
Both arms share exact expanded evidence manifests and effective-only index
schedule, slots16context/8time/4weak/4generated; source uniqueness inside each
family batch, least-used group/source/record selection, no replacement by easier
families on shortage. Report unique sources/groups, presentations and maximum
reuse. More repeated draws do not create more evidence.

Additionally8 fixed quality candidates/update from4 RM_fit generated windows,
2 candidates/window, cyclic over all16 candidates and both fit generators.
Use the same numeric inputs in both arms; A has zero quality-supervision weight.
This may require extra forward passes, recorded equally in both arms, with B
having additional head/gradient computation; do not claim exact equal FLOPs.
Stratify the candidate schedule by fit-only source/date/generator and coarse
fit-only quality quantiles; rotate rather than selecting only good candidates.

Save250/500/1000/2000/4000. Select each arm only on the common RM_cal macro
hard-task pair NLL, with predefined outputs for each family: context head,
time head, and each arm's generated quality score. Fit positive temperature on
cal after checkpoint selection, not on audit. Record task coverage and missing
families, and do not use development diagnostic to select checkpoints.

## Acceptance and independent value

Report per-head source/date-balanced C and D errors, rank correlation and
clear-Pareto pair accuracy; both in-generator and unseen-N1 strata. Compare
fixed first10/random10/top10 with source-window paired deltas, joint quality
improvement, diversity, qualified-candidate retention and each date group.
Separate GT-reading oracle heuristic from lawful RM selection. These remain
best-of16, full-window metrics, not official full-recording FRC/FRD or nativeK10.

Keep full temporal listener-only and balanced2x2 tests in the next evaluation;
the old summary-statistic Y-only50% is partly guaranteed by reciprocal pairing
and is insufficient evidence of conditional understanding. Multi-answer
retention is a metric/score proxy, not human recall or a mode-count ground truth.

A useful next result would be B improving same-source quality prediction and
fixed-pool utility over A without sacrificing both-qualified retention and
candidate differences. Current summaries justify this bounded diagnostic,
not integration into NFT. Independent fixed-P2 CCC/DTW guards remain required
for any separately approved future stage B.
