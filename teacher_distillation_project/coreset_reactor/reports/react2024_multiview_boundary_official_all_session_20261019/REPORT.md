# REACT2024 official-code, all-session-GT evaluation

## Protocol

This evaluation uses the Mam-Reactor metric implementation directly. The only
intentional protocol extension is the target pool: each prediction is matched
against every processed opposite-role GT recording from the same session.

- FRC: `framework.metrics.FRC._func`
- exact FRD: `framework.metrics.FRD._func`, including the official `tslearn.dtw`
- FRDiv: `framework.metrics.S_MSE.compute_s_mse`
- FRVar: `framework.metrics.FRVar.compute_FRVar`
- predictions use official AU rounding;
- GTs come from the deterministic official `Processor` cache;
- no neighbour matrix, sampled-ten subset, vectorized FRC substitute, or custom
  rolling-DTW implementation is used for the reported metrics.

The public official FRC/FRD wrappers parallel-map their corresponding `_func`
and take the mean. This runner invokes those same official `_func` bodies and
records their per-context outputs so the 2,248 context/arm evaluations are
resumable. A two-context check through the public `compute_FRC` and
`compute_FRD` wrappers matched the recorded `_func` aggregation exactly for
both arms.

The complete VAL1124 target pool contains 9–41 GTs per context, with a mean of
32.039146. The paired GT is placed first, followed by all remaining
opposite-role recordings from the same session; ordering does not affect the
best-match metrics.

## Full VAL1124 result

| Metric | Official baseline | Official `p=4` route | Delta |
|---|---:|---:|---:|
| FRC | 0.655348189611 | 0.655348189655 | +4.44e-11 |
| exact FRD | 86.025762868000 | 86.025762867522 | -4.78e-10 |
| FRDiv | 0.132445231080 | 0.132445231080 | 0 |
| FRVar | 0.058830551803 | 0.058830551803 | 0 |

The transferred route therefore has no measurable effect on REACT2024. All
6,340 eligible training inputs and all 1,124 validation inputs are exactly 750
frames. Multi-view training produces only one view at `start=0`, and powered
boundary fusion receives only one inference window. The tiny FRC/FRD deltas are
float32 multiplication/division roundoff from applying a weighted average to a
single window.

This is a matched validation result under the requested all-session extension,
not a hidden TEST claim. Genuine multi-view learning on REACT2024 would require
reconstructing sequences before the official 750-frame conversion, which would
change the evaluation unit and cease to be a matched REACT2024 comparison.

## Official source provenance

| Source | SHA-256 |
|---|---|
| `framework/metrics/FRC.py` | `f104754206490234847eea9ecc2a02218d9c4a41bc9fa88635c295023d1520b9` |
| `framework/metrics/FRD.py` | `0b01ae09383dfd7b36342186e8afc84f3a3b23b9990183bd654f70cfe12741b0` |
| `framework/metrics/S_MSE.py` | `0e4e9280ea676abf5ce34bbd78a303a93e9b3714e72fe88a169b47e0d57f4b40` |
| `framework/metrics/FRVar.py` | `e306cfe7a7d31572e86004ddd097804c5aec128624de646042775063dfe924f9` |
| `framework/modules/post_processor.py` | `b33b12035dc5002b10c4acc1ce586d8e9e1efdd01c07ade434f764ac6490ccd6` |

## Reproduction

```bash
PYTHONPATH=teacher_distillation_project .venv/bin/python -u \
  -m coreset_reactor.eval_react2024_official_all_session \
  --reference-dir teacher_distillation_project/coreset_reactor/reports/react2024_reference_count4_student_seed1 \
  --target-cache teacher_distillation_project/coreset_reactor/reports/react2024_mode_reference_distillation_seed1/val1124_gt_cache \
  --output-dir teacher_distillation_project/coreset_reactor/reports/react2024_multiview_boundary_official_all_session_20261019 \
  --power 4 --workers 48
```

Machine-readable aggregate results are in `results.json`; all official
per-context FRC/FRD values for both arms are in `official_rows.jsonl`.
