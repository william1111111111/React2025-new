# Current R1 versus native MAM

Same frozen Development80 K10 multi-target evaluation; exact FRD only fixed20-source subset. Each row uses the same checkpoint/native output policy across metrics. R1 continuous AU; MAM native rounding and pretrained budget differ. S-MSE/FRVar are not automatically quality improvements. R2 has not reached its first fixed evaluation checkpoint.

| Model | Step | FRC ↑ | FRD20 ↓ | S-MSE | FRVar |
|---|---:|---:|---:|---:|---:|
| MAM native archive | archive | 0.810962 | 172.576423 | 0.157204 | 0.058912 |
| R1 new head + C2 objective | 2000 | 0.589753 | 283.201032 | 0.138614 | 0.028756 |
| R1 new head + C2 objective | 6000 | 0.621484 | 208.176445 | 0.174397 | 0.054711 |

R1 has not beaten MAM jointly. Increased candidate spread does not establish appropriate/high-quality diversity; the source-level grouped dynamics are retained in task results. No new training decisions are inferred from this interim comparison.
