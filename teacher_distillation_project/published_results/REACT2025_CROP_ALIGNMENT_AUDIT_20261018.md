# REACT2025 center-window GT alignment comparison

Date: 2026-10-18 (Asia/Shanghai). Complete VAL571 replay; no training or new student inference.

## Controlled comparison

Frozen mode-reference basic student; the exact archived K10 predictions are reused. Both arms use the same 571 speaker-only VAL contexts, center750 source windows, all-session listener GT membership, AU rounding and official metric definitions. Before: complete GT is converted directly to the cropped prediction length. After: official Processor converts complete GT to complete source duration, then the identical source window is selected.

| Metric | Before (recomputed) | After | After − before | Relative change |
|---|---:|---:|---:|---:|
| FRC | 1.454372950 | 1.373898145 | -0.080474805 | -5.5333% |
| exact_FRD | 99.752944280 | 99.238433102 | -0.514511178 | -0.5158% |
| FRDiv | 0.170335159 | 0.170335159 | +0.000000000 | +0.0000% |
| FRVar | 0.069889866 | 0.069889866 | +0.000000000 | +0.0000% |

FRC decreases by 5.53%; lower-is-better exact FRD improves by 0.52%. Prediction-only diversity metrics are unchanged. These are evaluation-protocol effects, not a change in model ability.

## Verification

- All 571 historical GT cache keys reproduce from current file paths, order, sizes and mtimes. Raw GT file SHA-256 values are retained per context.
- Both legacy FRC and exact FRD were recomputed for every context and checked against the archived values. The same exact rolling-DTW implementation scores both arms.
- 457 contexts are cropped; 114 fit entirely inside 750 frames.
- All 563 equal-duration paired targets exactly equal the raw paired trajectory at the source window; 450 of these are cropped.
- Uncropped controls: maximum absolute FRC change 2.03e-08, maximum absolute FRD change 1.61e-06. These are floating-point-scale differences from regenerated postprocessing.
- FRVar replay differs from the archived JSON by about 7.5e-9 due to numerical reduction; before/after replay values are identical.
- 11 related tests passed, plus evaluator CLI and Python compilation checks.

## Interpretation and boundary

Same-session nonpaired GT is not annotated with frame-level semantic correspondence. Full-duration learned conversion followed by cropping is an explicit temporal convention, not verified event alignment or the only official convention. The official stochastic Processor is seeded per context in both arms; different duration conversions also change its computation and latent sampling path. This comparison measures the entire GT-policy change.

This remains speaker-only center750 all-session VAL evaluation. It does not resolve the separate full-sequence, bidirectional and sampled-ten differences from the released official test loader, and it is not an official hidden-TEST score.

## Use and artifacts

- New evaluator option: `--target-alignment full_then_crop`; separate cache version includes complete source length and crop offset.
- `legacy_compress` remains the default for frozen runners and reproducibility. Historical results and checkpoints were not overwritten.
- Full per-context scores: `coreset_reactor/reports/crop_alignment_audit_20261018/comparison.json`.
- Additional evidence: `gt_membership_audit.json`, `window_audit.json`, shard provenance/logs, cached target windows and source snapshots in that directory.
- Student SHA-256: `5fc1898b1594da7fe2b4642ad69e051187181de8a8df1432c7a31d4638e7ef1a`.
- Archived predictions SHA-256: `7c6ee611177d530faec124f91bbc9d0fe1f3c73050c1b48802b911a5e769cb84`.
