# Reward signal v2: fixed100 update comparison

This is a TRAIN signal-chain experiment, not a final FRC/FRD/diversity claim. R1 changes both feature scaling and kernel bandwidth; bandwidth-only and pooled-scale diagnostics were retained separately.

| Arm | Updates | Median coverage gradient | Median quality gradient | Median actual optimizer delta | Final probe output RMS change |
|---|---:|---:|---:|---:|---:|
| R0-baselined | 100 | 2.71016e-08 | 2.91327 | 0.00150884 | 0.000571866 |
| R1-calibrated | 100 | 0.409557 | 2.90934 | 0.00154094 | 0.000525754 |

coverage reward starvation reduced in this finite TRAIN comparison

All noise/source/target schedules, parent, policy initialization and optimization settings are shared. Native motion and action/parameter/covariance diagnostics are in probe_000/025/050/100.json. Old near-maximal speed/acceleration gates are not evidence of realistic motion.
