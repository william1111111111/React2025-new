# Reward policy training snapshot

2026-09-15T14:17:48.602005+08:00

| Arm | Updates / 500 | GPU / status |
|---|---:|---|
| R-quality | 500 / 500 | training complete |
| R-distance | 294 / 500 | 7 |
| R-coverage | 99 / 500 | 1 |

## First arm: R-quality

Completed 500 on-policy updates and 10000 candidate rollouts. Policy step500 is retained locally with a committed hash sidecar. The generator remains frozen.

Last 50 updates (100 TRAIN block episodes):

| Training metric | Policy | Independent parent reference |
|---|---:|---:|
| Best-target CCC, candidate mean | 0.055919 | 0.056339 |
| Full valid-block exact DTW, candidate mean | 8.082090 | 8.049753 |

Mean gated coverage: 0.018428; bad candidate rate: 0.000000. These are TRAIN-block metrics, not official FRC or full-record FRD.

Formal DEV80/exact FRD20 evaluation has not yet completed. No performance improvement is claimed. Action-change diagnostics found small updates; noisy finite-sample KL estimates can be slightly negative. Three arms share schedules/noise; GPU1 coverage parallelism was explicitly authorized after the initial serial launch.

Previous M0/M1 experiments are paused with step1700 checkpoints preserved. Source data, generated arrays and model checkpoints are excluded from Git.
