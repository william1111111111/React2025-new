# RM review: real evidence repairs and fixed-pool summary

No training, generator update or RL ran. Q0/Q1 and v1 artifacts were not modified.

## Real PTS recomputation

|Split|Old valid|New valid|Recovered|Lost|Valid sources|Date groups|
|---|---:|---:|---:|---:|---:|---:|
|RM_fit|56|255|199|0|137|17|
|RM_cal|9|41|32|0|24|4|
|RM_audit|12|103|91|0|54|3|

Counts come from original cached media PTS and hash-checked real listener arrays; synthetic tests are excluded. ±2seconds stays UNKNOWN. ±4seconds remains weak proxy supervision. Nearest-neighbor matching separately checks support, residual, continuity and large gaps. No endpoint clipping rescues out-of-range queries. All donor folds were verified.
Rejection reason counts overlap: a record can fail several rules. Not-evaluable activity/sync checks after a geometry failure are false, not independent evidence of inactivity. Detailed checks and signed residuals are in TEMPORAL_MAPPING.json.

## Effective sampling

Actual v1 logs in each arm:2601/4000 updates (65.025%) had zero time comparisons; total time presentations1691, mean0.42275/update. This is measured, not the earlier binomial estimate.
New prospective4000-step schedule draws8 effective time comparisons from8 distinct sources per batch, zero empty time batches. It contains32000 presentations of255 comparisons,137 sources,17date groups; this is repeated exposure, not32000 independent labels. UNKNOWN stays in the full index with zero weight.
Maximum temporal record reuse is941 under date-balanced scheduling; small date groups are repeatedly exposed. This is a reason to expand fit coverage before training, not a claim that sampling repairs evidence scarcity. Context/weak/generated valid counts remain154/319/2048. The sampler creates zero new independent labels.

## Full saved candidate selection

All864 records were summarized:3RMs x3generators x96windows. Group tables and both-qualified retention are in SELECTION_SUMMARY.json; GT-reading oracle heuristic is separately keyed. No candidates regenerated and no DTW recomputed.
RM-Gen on unseen N1 versus fixed random10: delta CCC sum -0.002434, delta native DTW sum -1.401983, delta S-MSE -0.002541;42.7% of windows improve both quality quantities. On the67windows with any double-qualified candidates, macro qualified retention rises55.60% ->68.58%. This is metric-defined retention, not human recall.
The3date groups disagree: one group loses both quality quantities (delta CCC -0.009081, DTW +0.016745), another improves both, and the third improves distance while losing CCC. RM-Gen on N0 versus random10 increases DTW by0.550130. There is no consistent joint-quality/diversity improvement.
These are best-of16 native-window sums, not full-recording FRC/FRD or nativeK10 inference. The metric oracle is a Pareto-front ordering heuristic, not a mathematical upper bound.

## Next decision

A bounded fresh-initialization A/B quality-prediction experiment is worth investigating after fit-only coverage expansion. A=repaired RM-Gen; B=same backbone plus separate CCC/DTW quality heads and direct fit-only numeric supervision. Detailed architecture, losses, bank regeneration cost and cal selection are in NEXT_EXPERIMENT_PLAN.md. Not started.
Do not treat old Y-only50% as sufficient conditional evidence: reciprocal pair construction enforces cancellation for marginal scores. Use stronger temporal listener-only and balanced interaction diagnostics.
The previously examined audit set is now a development diagnostic. No fresh independent confirmation is claimed. No evidence here warrants connecting the current RM as the sole NFT judge.

## Validation

9necessary tests passed (nearest PTS, real-support handling, missing frames, residuals, UNKNOWN/fold sampling, padding/order, candidate dimensions and source gradient). Real recomputation checked donor folds and immutable trajectory hashes. No new model training.
