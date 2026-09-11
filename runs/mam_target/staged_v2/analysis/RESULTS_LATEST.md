# Staged quality-protected refinement

Development-80, seed123, native continuous AU, K10. Exact FRD belongs to the frozen 20-source subset; only complete 2000-pair results are reported.

| Model | Total steps | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|---:|---:|
| MAM archive (native rounding; different training budget) | archive | 0.810962319 | 0.766297889 | 172.576423473 | 0.157203704 | 0.058911592 |
| R2 parent | 6000 | 1.181468685 | 1.134674340 | 133.409074259 | 0.043679938 | 0.039064668 |
| S0_quality | 6500 | 1.206096718 | 1.156380026 | pending | 0.042919219 | 0.039343890 |
| S0_quality | 7000 | 1.207257839 | 1.154813395 | pending | 0.043443628 | 0.040269297 |
| S0_quality | 8000 | 1.221969005 | 1.151122761 | 133.312706288 | 0.043695860 | 0.040877711 |
| S1_joint | 6500 | 1.190077312 | 1.140435098 | pending | 0.047142901 | 0.040200230 |
| S1_joint | 7000 | 1.195026556 | 1.137634576 | pending | 0.048823800 | 0.042002674 |
| S1_joint | 8000 | 1.202744548 | 1.134012060 | 135.929274782 | 0.050737489 | 0.042962462 |
| S2_staged | 6500 | 1.188429487 | 1.141298879 | pending | 0.043158367 | 0.039239138 |
| S2_staged | 7000 | 1.193576048 | 1.148268104 | pending | 0.047238994 | 0.041586705 |
| S2_staged | 8000 | 1.195617702 | 1.140947380 | 133.822531509 | 0.048848547 | 0.041989461 |

Baseline audit and original fingerprints: ../refine_v1/RESULTS_LATEST.json (read-only).

S0 includes the common score guard; it is not unchanged historical R2 continuation. S1−S0 tests the dispersion term, S2−S1 tests the update schedule. No task advantage is inferred from training losses alone.

Final selection awaits the +2000 endpoint and complete FRD20 for each arm. The quality band is a development engineering criterion, not equivalence. Prior A/B remains paused; no additional seeds or architectures are launched.
