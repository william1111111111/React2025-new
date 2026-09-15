# Mode supervision progress snapshot

Snapshot: 2026-09-15T12:18:08.501762+08:00

Training is ongoing; these are intermediate 1,000-step DEV80 results, not final 6,000-step joint results. exact FRD20 has not run yet.

| Arm | Decoder updates at snapshot | FRC80 at 1000 | S-MSE80 at 1000 | FRVar80 at 1000 |
|---|---:|---:|---:|---:|
| M0-control | 1558 | 0.903622276 | 0.061120026 | 0.045097332 |
| M1-mode | 1555 | 0.883704233 | 0.057296108 | 0.038766954 |

M1 has not improved FRC or diversity over M0 at this intermediate checkpoint. Both arms continue their fixed budget on GPU7.

Recorded incidents: one-frame task-loss failure (fixed with equal masked handling); later unexplained process disappearance (resumed from step500); monitor timestamp collision (fixed and tested). See tail_fix_amendment.json and experiments/mode_runtime/README.md.

Full-resolution clock caches/cohort are rebuildable local files excluded from Git; their identities and preparation code remain recorded. Model checkpoints and prediction arrays remain local.
