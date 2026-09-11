# Staged quality-protected refinement

Development-80, seed123, native continuous AU, K10. Exact FRD belongs to the frozen 20-source subset; only complete 2000-pair results are reported.

| Model | Total steps | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|---:|---:|
| MAM archive (native rounding; different training budget) | archive | 0.810962319 | 0.766297889 | 172.576423473 | 0.157203704 | 0.058911592 |
| R2 parent | 6000 | 1.181468685 | 1.134674340 | 133.409074259 | 0.043679938 | 0.039064668 |

Baseline audit and original fingerprints: ../refine_v1/RESULTS_LATEST.json (read-only).

S0 includes the common score guard; it is not unchanged historical R2 continuation. S1−S0 tests the dispersion term, S2−S1 tests the update schedule. No task advantage is inferred from training losses alone.

Final selection awaits the +2000 endpoint and complete FRD20 for each arm. The quality band is a development engineering criterion, not equivalence. Prior A/B remains paused; no additional seeds or architectures are launched.
