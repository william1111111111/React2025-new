# Stage A: conditional matching reward models

All evidence is automatic/data-supported, not human preference. No generator update or RL.

| Arm | Selected step | Context accuracy | Active time accuracy | Unseen-generator accuracy | Weak retention |
|---|---:|---:|---:|---:|---:|
| RM-Paired | 1000 | 0.62 | 0.5833333333333334 | 0.49612403100775193 | 0.0947 |
| RM-Multi | 1000 | 0.54 | 0.3333333333333333 | 0.5219638242894057 | 0.1895 |
| RM-Gen | 1000 | 0.68 | 0.3333333333333333 | 0.5271317829457365 | 0.0772 |

Scope: 416 fixed TRAIN windows, recording-date grouped split. Exact candidate-window DTW is not full-recording FRD. Each fixed-pool selection consumes 16 generated candidates.
Independent listener-only and balanced interaction diagnostics are under probes/. Full-recording aggregation and weak-answer retention are under audit/.
Mode acceptance remains a score-relative diagnostic: no independent human or mode truth was created.
