# Reward signal repair (fixed 100 updates)

This independent experiment preserves P2, K10, TRAIN-only rewards and historical artifacts. No official evaluation or teacher inference is rerun in this phase. M0/M1 remain paused. No automatic push, learning-rate change, KL removal, architecture change or added seed.

A: `EXISTING_DATA_AUDIT.json` reuses all 64 calibration groups, 16 reachability groups with16 independent references, and1,000 independent reference tables. Old per-candidate *policy* reward matrices/predictions were not saved, so these tables must not be called a full policy history. Fixed old-policy replays live in `phase_a_replay/`; their gradients are explicitly B1-recomputed, while original-estimator gradients remain in the old archived KL diagnostic. These replays do not update weights.

B1: `rewards.py` uses the other K-1 candidates' mean quality as an action-independent baseline, keeps original K normalization, and computes per-target coverage differences separately. Tiny coverage credit is not subtracted between two large mixed reward sums. Tests cover an exactly enumerable score expectation, nonzero1e-200 marginal credits, and valid-mode distance masks.

B2: deterministic TRAIN fit/hold partition is frozen in PREFLIGHT.json (40fit/24hold). Three numerical artifacts are retained, not hidden:
1. BANDWIDTH.json keeps old geometry and changes only h2. Individual-mode checks passed but audit found79% contribution from one physical transformed AU channel across its four modes, so it was not used for formal training.
2. GEOMETRY.json fits pooled generated+real IQR. A10:4 mixture can hide minority real-target variability from the pooled IQR; it was not used for formal training.
3. GEOMETRY_STRATIFIED.json explicitly uses max(generated IQR, target IQR, pooled IQR), then the same predeclared group floor and bandwidth rule. R1 uses this version. No DEV score or coordinate deletion was used. AU contributions remain dominant and this limitation must accompany interpretation.

C: exactly100 identity-initialized updates per arm, first200 original training episodes, same original noise/reference schedule, AdamW1e-5, KL .01, clip1. R0 uses B1+old geometry/kernel; R1 uses B1+stratified geometry/calibrated bandwidth. Consequently this is a geometry-and-bandwidth repair comparison, not a bandwidth-only causal claim. Both run concurrently on GPU0 within previously authorized resources.

At0/25/50/100, fixed independent TRAIN probes record parameter hash/change, masked action covariance, coupling shift/scale, full and rewarded-block noise deltas, native prediction deltas and real motion quantiles, reward credits, and separate gradients. Every actual training update records the true AdamW parameter delta; clipping multiplier is not treated as optimizer-step multiplier. Negative finite-sample KL estimates are retained.

Costs:2,000 action rollouts per arm plus80 fixed-probe rollouts; existing reference draws are reused with their historical generation cost retained. Old-policy phaseA adds120 diagnostic action rollouts. No claim of improved diversity or preserved quality is made without formal same-checkpoint joint evaluation, which is deliberately not scheduled in this finite signal-chain phase.
