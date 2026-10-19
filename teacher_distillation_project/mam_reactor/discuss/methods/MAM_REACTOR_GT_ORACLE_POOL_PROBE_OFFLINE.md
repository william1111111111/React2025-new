# Mam-Reactor GT-Oracle Candidate-Pool Probe

## Research question

Does the current Mam-Reactor already expose alternative deterministic candidate
trajectories that increase official FRDiv while preserving the complete
10-candidate set's FRC and exact FRD?

This is an oracle feasibility probe on the train-side B_fit split. Listener GT
is used to score/select candidates. It is **not** a deployable selector and is
**not** a test-set score.

## Candidate pool

The original q0--q9 are always retained. Extra deterministic candidates come
from the same checkpoint:

- dynamic residual: 0.80×, 1.20×;
- static-style residual: 0.75×, 1.25×;
- q1--q8 constant gate: 0.75, 1.00;
- q9 uniform mixture;
- q9 eight one-hot emotion mixtures.

Default baseline inference contract:

- residual scale: 0.95;
- style scale: 0.95;
- AU multiplier: 1.8;
- VA multiplier: 0.4;
- expression multiplier: 1.6.

## Quality labels

For each candidate:

`c_i = max_j CCC(pred_i, GT_j)`

`d_i = min_j [(1/15)DTW(AU) + DTW(VA) + (1/8)DTW(expression)]`

The exact rolling-DTW kernel is imported from
`tools/compute_frd_resumable.py`.

## Selection contract

Start from the exact ten baseline predictions. q0 is fixed. Each q1--q9 slot
may only be replaced by another candidate from that same query slot.

Maximize FRDiv subject to, per context:

`sum(selected c_i) >= sum(baseline c_i)`

`sum(selected d_i) <= sum(baseline d_i)`

First run uses zero slack. One-swap and two-swap local search are allowed.

## Decision

A positive result means the generator already contains exploitable
quality-preserving diversity headroom and justifies a later GT-free selector
distillation stage. If the strict oracle finds almost no headroom, candidate
generation—not allocation—is the next target.
