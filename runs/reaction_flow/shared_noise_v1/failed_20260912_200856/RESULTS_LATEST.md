# Shared/local noise: actual results

Native continuous AU, FP64 Euler16, Development80/K10; exactFRD20 only when all2000 pairs complete. Inherited T0-14000 and R2 encoder warmstart costs apply.

| Model | FRC80 | exactFRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| MAM native archive | .810962319 | 172.576423473 | .157203704 | .058911592 |
| T0 parent14000 | .859622576 | 133.876667407 | .081215642 | .053756177 |

No missing endpoint is filled with prior-model metrics. Compare DC/slow/fast and block covariance jointly with quality and source/noise-factor probes; no DC value is a required human variance target. MAM native AU/budget differ. Single-seed repeated development evidence, not independent confirmation.
