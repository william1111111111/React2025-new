# Fixed-P2 quality-guarded NFT

Q0-quality-guarded and Q1-coverage-guarded start at the same original P2,
seed 123, with the complete reaction velocity trainable. Source condition,
BERT, semantics and reference are frozen. Historical NFT assets are read-only.

Absolute fixed-reference quality feedback is not group-centered. Coverage is
positive only for supported candidates in groups satisfying both quality gates.
Each proposal additionally differentiates an independent TRAIN K4 Euler16
pure-noise rollout. CCC and native hard-DTW active-path losses have separate
fixed weights mu_C=mu_D=1. These losses compare task scores, not P2 trajectories.
Native DTW paths are recomputed for each output and each attribute group;
the complete three-group sum selects a single target. No 32-point proxy is used.

Sixteen fixed complete TRAIN recordings control acceptance every 100 proposals.
Their target pools, Processor outputs, source noise and P2 scores are frozen.
Acceptance requires FRC >= P2 and FRD <= P2, with zero budget relaxation.
This TRAIN monitor is not an independent validation set. Rejection restores
student, optimizer, old and RNG; next proposals use new scheduled IDs/noise.
Intermediate 20-step old refreshes are tentative and roll back with the
transaction. Stop after 3 consecutive rejected transactions or 1000 proposals.
The learning-rate warmup follows proposal count, not accepted count.

Gradient component diagnostics run on proposal 1 and every 20 proposals;
other entries are absent/null rather than invented zero measurements.
All updates record actual parameter movement and the two independent violations.
Synthetic tests and real TRAIN checks are stored separately from formal results.
Preflight generated 12 student + 12 reference quality trajectories and 2 old
endpoints. No preflight model state initializes the formal runs.

Run `python -m generator_quality_guard.queue` after preflight. It freezes code
and input hashes, runs both arms on GPU 0, then evaluates last-proposal and
last-accepted checkpoints separately using official DEV80 and exact FRD20.
Unchanged P2 results may be reused only after weight equality and artifact hashes
are verified. No DEV/TEST result controls training. Stopping at unchanged P2
is not evidence of a learned diversity improvement. No automatic push.

Outputs: `runs/reaction_flow/generator_quality_guard_v1/`.
