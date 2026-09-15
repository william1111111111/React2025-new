# Reward signal v2: joint evaluation

Both step100 policies use the fixed P2 generator, source-only K=10 inference. FRC and diversity use full DEV80; exact FRD20 completes 2000 pairs per arm. P2 is the previously completed reference evaluation.

| Model | FRC | exact FRD20 | S-MSE | FRVar |
|---|---:|---:|---:|---:|
| P2 reference | 0.937901101 | 130.688674740 | 0.067720607 | 0.055246130 |
| R0-baselined | 0.937600936 | 130.680603197 | 0.067707568 | 0.055230834 |
| R1-calibrated | 0.937571132 | 130.680369493 | 0.067711584 | 0.055233259 |

The calibrated reward supplies stronger coverage gradients, but this 100-update comparison shows no clear task-level benefit: diversity is nearly unchanged, FRC slightly lower, and FRD slightly lower. These small differences are not a statistical significance claim. R1 changes both feature scales and bandwidth; it is not a bandwidth-only ablation.
