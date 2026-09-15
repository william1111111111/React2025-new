# Quality-constrained explicit sampling policy

Independent experiment; M0/M1 paused with step1700 checkpoints preserved. Frozen generator: P2-BERT step1000, SHA 780d761e9efd7bfbc5a32dc12009928cc1b686e6df90d92c3b773b886a1d9986. Its complete evaluation is locked in PROTOCOL.json. No generator or BERT training, no paid API calls.

The 96D action consists of four orthogonal actual-PTS low-frequency coefficients of each candidate's original full-record Gaussian noise. Four alternating conditional affine couplings start exactly at identity; public generation modifies only those noise coefficients and retains the original solver, candidate keys, source-only semantic inputs and postprocessing. Each candidate samples once per recording, not per block. Missing legal TRAIN semantic caches are NULL, not listener-derived content.

Separate arms: R-quality, R-distance and R-coverage. All share initialization, 1,000 fixed TRAIN episodes (two per update), base action noise and independent reference tables. Each performs 500 updates / 10,000 candidate rollouts. Actual unsmoothed DTW uses the full valid block; this is explicitly a TRAIN-block reward, not official full-record FRD. Short blocks with fewer than two paired frames are excluded from reward episodes, not interpolated to fake CCC support. Coverage gates CCC and DTW on the same candidate-target pair.

Difference advantages use original K normalization without candidate centering. Reward actions are detached for the true joint action log probability; the independent reparameterized KL sample retains its gradient. Dual coefficients, EMA, policy, optimizer, RNG and progress are checkpointed. Total KL >20 triggers a fixed stop, not threshold relaxation. No PPO reuse, FM pseudo-likelihood or GT selection at inference.

Before formal updates: 64 TRAIN calibration groups, 16×32 reachability candidates plus independent references, real initialization equality, density/Jacobian/padding tests, enumerable score-gradient check, and real resume/gradient tests. These extra pilot costs are not counted as the arms' formal 10,000 rollouts. The fixed reachability gate is in PROTOCOL.json; no launch if it fails.

`queue.py` runs serially on GPU7 because other jobs occupy most of that card. It records child PIDs, return codes/signals, log paths and 30-second heartbeat. Checkpoints at100/500; full DEV80 and exact FRD20 are run at500. The original sampler is a zero-training reference. No automatic reward grids, whole-Flow RL, budget extension or push.

Outputs: runs/reaction_flow/reward_policy_v1. Frozen historical result files are never overwritten. Final claims require the same policy500 + fixed generator combination and complete exact FRD20, not synthetic verification_results.json supplied in the attachment.
