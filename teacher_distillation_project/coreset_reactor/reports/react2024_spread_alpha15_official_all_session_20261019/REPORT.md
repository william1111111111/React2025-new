# REACT2024 slot-spread scaling under official all-session evaluation

## Protocol

- Split: full REACT2024 validation set, 1,124 contexts.
- Target pool: every processed opposite-role GT recording from the same session,
  with the paired target first (9--41 GTs per context, mean 32.0391).
- Quality metrics: the unmodified Mam-Reactor
  `framework.metrics.FRC._func` and `framework.metrics.FRD._func`, averaged over
  contexts exactly as in the official wrappers.
- Diversity metrics: the unmodified official `compute_s_mse` and
  `compute_FRVar` implementations.
- Output contract: AUs are rounded before all four official metrics.
- Transform: for the ten prediction slots at every frame,
  `y_k = mean(x) + 1.5 * (x_k - mean(x))`, followed by valid-range clipping
  and expression-probability renormalization.

## Results

| Method | FRC ↑ | Exact FRD ↓ | FRDiv ↑ | FRVar ↑ |
|---|---:|---:|---:|---:|
| Baseline | 0.655348190 | 86.025762868 | 0.132445231 | 0.058830552 |
| Slot spread, alpha=1.5 | 0.703201093 | 93.945012815 | 0.168620512 | 0.071231730 |
| Absolute change | +0.047852903 | +7.919249947 | +0.036175281 | +0.012401178 |
| Relative change | +7.3019% | +9.2057% (worse) | +27.3134% | +21.0795% |

The scaling clearly exposes differences among the ten reaction candidates:
FRC, FRDiv, and FRVar all rise substantially. It also moves their distribution
farther from the GT distribution, so exact FRD becomes 9.21% worse. Therefore
alpha=1.5 is a quality/diversity tradeoff rather than a uniformly better
operating point.

This transform remains effective on REACT2024 even though every accepted clip
is exactly 750 frames, because it scales differences between prediction slots.
The powered overlap method scales temporal windows and is an identity when
there is only one 750-frame window.

## Reproducibility

The complete machine-readable result, source hashes, data hashes, and per-row
official scores are in `results.json` and `official_rows.jsonl`. The baseline
rows were reused from the preceding run with the same prediction cache, GT
cache, and official metric sources; overlapping recomputed rows were required
to match exactly. All 1,124 alpha=1.5 rows were computed in this run.

This is a validation result, not a hidden-test claim.
