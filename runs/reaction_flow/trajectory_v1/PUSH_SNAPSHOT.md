# Push snapshot: completed training, ongoing evaluation

Flow A completed 12000/12000 updates (seed123). Frozen condition encoder inherits R2-6000 training cost; no MAM weights. Production evaluation uses uniform FP64 Euler16 with FP32 inputs to unchanged legacy metrics, as recorded in numerical_protocol_fp64_v3.json.

| Checkpoint | FRC80 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|
| seed123_A_step2000 | 0.360733617598 | 0.106513001025 | 0.059495832771 |
| seed123_A_step6000 | 0.373638848665 | 0.140510275960 | 0.071861959994 |

A12000 evaluation remains in progress; no final Flow exact FRD20 result exists at this snapshot. No F-cont/F-task training has started. Early Flow FRC remains below archived MAM and R2 despite larger candidate spread. Six flow regression tests passed; legacy FP64-to-FP32 interface probe passed.

Completed S0/S1/S2 total8000 and their exact FRD20 results are included separately under runs/mam_target/staged_v2. Removed obsolete FP32 export arrays are enumerated in cleanup_superseded_exports.json; historical metrics/checkpoints retained locally.
