# Completed task/dynamics adaptation

Both seed123 endpoints exceed archived MAM FRC80 and exact FRD20 on their respective frozen development populations. Candidate S-MSE remains approximately half the MAM reference; the full quality/diversity objective is not yet achieved. T1 slightly improves FRD and spread relative to T0 but has lower FRC, so no clear joint advantage of the dynamic term is established.

| Model | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| MAM native | 0.810962319 | 172.576423473 | 0.157203704 | 0.058911592 |
| A12000 | 0.414712854 | 154.005436547 | 0.139438540 | 0.068698995 |
| T0-task | 0.859622576371 | 133.876667406820 | 0.081215642393 | 0.053756177425 |
| T1-task-dynamics | 0.833056843404 | 133.433357504613 | 0.082931809127 | 0.052843958139 |

Each adaptation row is one total14000 checkpoint, continuous AU, FP64 Euler16 sampling followed by FP32 unchanged official metrics. FRD20 is the fixed20-source subset, all2000 candidate-target pairs complete, not full Development80 FRD. MAM retains native AU rounding and its different training budget. These are reused development results, not independent confirmation or multi-seed evidence.

Each arm inherited A12000 and its AdamW state, changed lr to2e-5, and completed2000 new steps with identical schedule and frozen condition encoder. A additionally inherits6000 R2 training updates through its encoder. Actual per-arm training took about5.54h in parallel, each exposing5702765 valid endpoint frames. Checkpoint and result hashes are in FINAL_RESULTS.json. No extra seed or budget was launched.

Diagnostics and limitations: P8 decomposition and four-source Euler16/32/64 probes are recorded separately; more solver steps did not show consistent convergence. TRAIN task precision discrepancy was1.19e-7. Ten tests passed; actual T750 short resume parameter error was0 with equal optimizer state. All training/evaluation commands and raw logs are retained. RESULTS_PROGRESS.md is an earlier progress snapshot, superseded by this completed summary.
