# Reaction reward model, stage A

Standalone fresh 256-wide dual-encoder conditional judge. No P2 task encoder
weights are loaded by the judge. P2/NFT appear only in the frozen candidate-bank
process, using NULL text and native Euler16 standard-noise outputs.

Pipeline: prepare -> evidence -> generated_bank -> generated_evidence -> three
4000-step RM arms -> calibration/selection -> audit -> unimodal/2x2 probes ->
weak-answer/full-recording diagnostics -> report. The queue never starts RL.

Data boundaries: all raw media and input feature hashes are inventoried. Because
no verified person/dyad map was available, all same-date recordings across every
session are conservatively kept together, with exact duplicate closure. This
binds both role directions and all subsequent crops. The split is explicitly
recording-date-disjoint, not participant-disjoint; date provenance is an explicit
assumption, not identity verification. Only RM_fit determines normalization and
activity threshold. Donors must share the source's fold. Actual counts are
1100/265/295 over 24 date components, rather than pretending exact 70/15/15.

UNKNOWN is preserved and excluded from gradients. ±2-second shifts remain
UNKNOWN because natural delay ambiguity is unresolved. ±4-second shifts require
native contiguous PTS support, active source/response and measurable change;
they are weak temporal proxies, not human preference truth. Same-session targets
have no time-head supervision. No circular shifts or asymmetric padding.

All three arms share initialization and base/time sampling and dropout seeds.
Family loss masks preserve missing slots. Optimization is AdamW, lr1e-4,
warmup200, clip1, seed123. Checkpoints1000/2000/4000 are selected by RM_cal
macro pair NLL; RM_audit is only scored after selections are frozen.

Candidate-bank cost <=14848 windows. Native full-window DTW is used for generated
Pareto labels; it is NOT official full-recording FRD. Same-generator comparisons
constitute at least half the retained preferences. N1 is audit-only. Historical
candidate generators may have trained on all TRAIN groups: judge group holdout
is not generator data holdout.

Limitations: short/neutral/ambiguous evidence is often UNKNOWN; audit temporal
coverage is small and will be reported as such. Full-recording RM diagnostic
uses length-weighted <=750-frame chunks. X-only cancels algebraically for
same-source pairs and is reported as a sanity check, not a learned achievement.
An independent Y-only probe is trained. Acceptance is reference-relative proxy
retention, not probability of correctness. There is no independently validated
mode taxonomy; no mode-count truth is invented. Final decision remains subject
to these limitations and no code path automatically starts stage B.

Run output: runs/reaction_reward/v1. Old quality-guarded NFT remains paused.
