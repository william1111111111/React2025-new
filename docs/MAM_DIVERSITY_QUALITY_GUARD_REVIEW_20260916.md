# Local MAM-Reactor diversity audit, 2026-09-16

Scope: local `/home/zhengshiyi/react2025/mam_reactor` implementation and archived
configuration; these findings do not establish settings used by an external paper.
No running Q0/Q1 code, acceptance rule, or historical result was changed.

## Verified controls

`results/offline_evidence/config.json` records channel_diversity_budget=0.15,
channel_diversity_weight=20, official_diversity_batch_weight=20,
official_diversity_margin=0.15. In regnn/emotion_query_mamba.py:2756 the batch
penalty is relu(margin - generated set diversity). Lines 2760–2820 additionally
match AU/VA/expression diversity contributions to TRAIN target shares, with
per-source desired total min(target diversity, 0.15). Thus 0.15 is a capped
training budget, not a guaranteed achieved score or a demand for each source.

Training config records residual/style scales .95, AU multiplier1.8, VA .4,
expression1.6. Reproducibility commands instead give VA .5; settings must be
attributed to their specific artifact, not merged into one claimed run.
The architecture has a frozen q0 anchor and explicit query residuals, so these
multipliers cannot be transplanted as Flow noise temperatures.

The MAM diversity surrogate uses straight-through AU rounding and common T*25
denominators across channel groups. Any adaptation must first verify our actual
output/postprocessing convention, not silently alter official metrics.
An isolated VA calibration updates only 258 scalars and uses preservation terms;
it is a separate intervention, not evidence for multiplying all output variance.

## Current Q1 diagnostic snapshot

528 saved groups: coverage enabled189 (35.8%); eligible candidate-target pairs
14.61%; mean candidate nontrivial coverage gain fraction13.66% (before final
whole-group coverage gate). Coverage is a marginal target coverage reward, not
an explicit S-MSE target. These are mechanism diagnostics, not proof of the
cause of diversity decline or final method performance.

## Recommended bounded next comparison

Keep the existing run intact. In a new registered comparison, retain full
velocity training and source-only sampling, add one target-supported,
channel-balanced diversity-budget term on the existing differentiable pure-noise
K4 rollout. Compare against the identical model without this term, with shared
initialization, examples and noise. Calibrate its gradient scale on TRAIN rather
than copying MAM's weight20. Report AU/VA/expression, DC/slow/fast and motion
speed/acceleration so static separation or jitter cannot masquerade as success.
K4 block geometry is a training estimate; final native K10 DEV80/exactFRD20
must satisfy the user's FRC>=.94, FRD<150, S-MSE approximately .15 together.
TRAIN quality allowances need separate prespecified calibration; DEV thresholds
must not simply be substituted into differently scoped TRAIN block scores.
The strict P2 guard may remain binding: adding a diversity objective alone does
not guarantee reaching .15. No broad grid, inference scaling, or new run launched.
