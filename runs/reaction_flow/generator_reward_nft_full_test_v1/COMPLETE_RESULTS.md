# Complete local TEST evaluation

Local TEST is NOT an independent holdout: all 3426 audited input/target feature files are byte-identical to VAL. Checkpoints were fixed before this run; results do not tune training.

Official directional population; original speaker and reverse directions reported separately. K10 native source-only; all TEST semantic inputs NULL because no split-legal TEST cache exists. Full exact FRD, not FRD20.

| Model / group | N | FRC | exact FRD | S-MSE | FRVar | temporal S-MSE | TLCC | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 / full_local_TEST | 1142 | 0.786617945 | 127.926324864 | 0.066014679 | 0.053226778 | 0.056890124 | 47.853765324 | 0.187993852 |
| P2 / original_speaker_direction | 571 | 0.988703947 | 119.723068871 | 0.066830314 | 0.056529287 | 0.057803410 | 47.492119089 | 0.180719499 |
| P2 / reverse_direction | 571 | 0.584531942 | 136.129580858 | 0.065199045 | 0.049924269 | 0.055976838 | 48.215411559 | 0.195268206 |
| N0-quality / full_local_TEST | 1142 | 0.637860396 | 140.072294613 | 0.117397092 | 0.062714814 | 0.101564480 | 47.638353765 | 0.199269166 |
| N0-quality / original_speaker_direction | 571 | 0.788191024 | 135.884831185 | 0.121485911 | 0.066288117 | 0.105222883 | 47.374781086 | 0.193673079 |
| N0-quality / reverse_direction | 571 | 0.487529769 | 144.259758040 | 0.113308273 | 0.059141510 | 0.097906076 | 47.901926445 | 0.204865253 |
| N1-quality-coverage / full_local_TEST | 1142 | 0.642671992 | 138.311842315 | 0.110829685 | 0.060619132 | 0.096297279 | 47.771453590 | 0.198580886 |
| N1-quality-coverage / original_speaker_direction | 571 | 0.800968649 | 133.492067048 | 0.115083320 | 0.064110706 | 0.099992732 | 47.481611208 | 0.192765153 |
| N1-quality-coverage / reverse_direction | 571 | 0.484375334 | 143.131617583 | 0.106576051 | 0.057127558 | 0.092601825 | 48.061295972 | 0.204396619 |
