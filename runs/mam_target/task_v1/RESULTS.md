# MAM-target first batch actual status

Queue completed: True; failures: []. Single seed123 Development80; FRD20 is subset only.

| arm | step | FRC ↑ | exact FRD20 ↓ | S-MSE | FRVar |
|---|---:|---:|---:|---:|---:|
| R1 | 2000 | 0.589753 | 283.2010320836316 | 0.138614 | 0.028756 |
| R1 | 6000 | 0.621484 | 208.17644542251006 | 0.174397 | 0.054711 |
| R2 | 2000 | 1.140675 | 135.15241635093878 | 0.023093 | 0.028714 |
| R2 | 6000 | 1.181469 | 133.40907425912587 | 0.043680 | 0.039065 |
| MAM native archive | archived | 0.810962 | 172.576423 | 0.157204 | 0.058912 |

Do not mix best FRC and best FRD from different checkpoints. MAM pretraining/rounding/budget differ. Diversity requires grouped trajectory checks stored with exports, not maximizing variance. No flow/codec or additional seeds automatically started. Raw commands, checkpoints, identities and task matrices retained.
