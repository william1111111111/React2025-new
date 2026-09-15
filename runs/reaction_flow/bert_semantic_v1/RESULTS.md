# Frozen-BERT semantic controls

Status: complete. All main results use correct source-only inputs.

| Arm | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| P0-null | 0.937152 | 130.986103 | 0.069084 | 0.056325 |
| P1-byte | 0.912151 | 129.571334 | 0.068478 | 0.053893 |
| P2-bert | 0.937901 | 130.688675 | 0.067721 | 0.055246 |
| P3-bert-time | 0.932038 | 130.241970 | 0.068285 | 0.053571 |

RESULTS.json contains source-paired deltas, fixed-cache NULL/non-NULL/timed strata, session-equal summaries and session bootstrap intervals, target-best CCC coverage, DC/slow/fast decomposition, parameter/resource counts and separate fixed interventions. FRD20 is the fixed 20-source subset, all2000 pairs; it is not full80 FRD.

Limits: 48 TRAIN sources repeated over 1000 updates, single seed123; offline event text, no online/causal claim; TRAIN/DEV proposal distributions remain different. P2-P1 changes pretrained resources and representation architecture. Byte768 is not the historical byte256 model. P0 instantiates fusion but does not use semantic content; active parameter counts differ. BERT features are deterministic across K and do not directly supply new random modes. No event type or planner is used. Unknown roles and missing inputs remain NULL, not no-event.

Next decision: Review P2-bert against the fixed text/time interventions before widening TRAIN coverage; passing tolerances is not a statistical superiority claim.
