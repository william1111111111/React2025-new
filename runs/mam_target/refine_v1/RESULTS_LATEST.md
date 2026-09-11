# Latest verified baseline results

| Model | step | FRC80 ↑ | FRC20 ↑ | exact FRD20 ↓ | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|---:|---:|
| MAM | archive | 0.810962318916 | 0.766297889032 | 172.576423472971 | 0.157203704119 | 0.058911591768 |
| R1 | 2000 | 0.589753361808 | 0.571019078487 | 283.201032083632 | 0.138614177704 | 0.028756171465 |
| R1 | 6000 | 0.621483527858 | 0.603919672327 | 208.176445422510 | 0.174397036433 | 0.054711170495 |
| R2 | 2000 | 1.140675347579 | 1.122607729902 | 135.152416350939 | 0.023093234748 | 0.028714386746 |
| R2 | 6000 | 1.181468684930 | 1.134674339995 | 133.409074259126 | 0.043679937720 | 0.039064668119 |

All exact FRD scores verified from2000/2000 cached pairs each. Identity hashes link FRD to the SAME task exports and checkpoint. No recomputation. R2 step6000 is the fixed common refinement parent. Native AU policies differ from archived MAM; all data are development, not independent confirmation.

Detailed channel/coverage diagnostics are in baseline_channel_summary.csv and baseline_per_input.csv. A matched target SLOT is not a semantic mode; duplicate target content slots are explicitly counted.
