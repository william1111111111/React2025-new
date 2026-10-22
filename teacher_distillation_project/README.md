# Teacher Distillation Project

This project contains the REACT2024/REACT2025 fixed-teacher, mode-reference
teacher, expert-student distillation, evaluation and calibration route.

## Layout

- `coreset_reactor/`: canonical research code, local reports and frozen weights.
- `mam_reactor/`: the model/evaluation runtime required by the distillation code.
- `data`, `datasets`, `runs`: links to the shared datasets and run storage.
- `.venv`: link to the verified shared Python environment.
- `published_results/`: small, human-readable summaries of accepted results.

Large reports and weights remain local and are ignored by Git. The copied
Mam-Reactor runtime has its own source tree. Large immutable assets are
hard-linked, so they remain available if the old compatibility links are
removed without consuming a second full copy of the storage blocks.

## Use

```bash
cd /home/zhaowenjie/react2025_new/teacher_distillation_project
source env.sh
./verify_project.sh
$REACT_PYTHON -m coreset_reactor.run_mode_reference_student --help
$REACT_PYTHON -m coreset_reactor.run_react2024_mode_distillation --help
```

The accepted REACT2024 operating point is the count4 student followed by a
global slot-centered spread calibration with alpha 1.50. It is a deliberate
quality/diversity trade-off: full VAL1124 all-session evaluation gives FRC
0.703201, exact FRD 93.945013, FRDiv 0.168621 and FRVar 0.071232. The original
uncalibrated checkpoint and results are retained.

REACT2025 is not changed by that calibration decision.

The 2026-10-18 REACT2025 long-sequence experiment adds deterministic
head/center/tail teacher views for training clips longer than 750 frames and
50% overlap, center-weighted full-sequence inference. Its original forward
VAL571 tables are retained only as diagnostics. Paper ablations now require
full-sequence VAL1142, both directions, all same-session GT, and all five
repository metrics; see `coreset_reactor/EVALUATION_CONTRACT.md`.

Under that protocol, the current multi-view student with normalized `w^4`
fusion and the final query-0 constant-AU transition postprocess gives FRC
1.101718, exact FRD 136.159593, FRDiv 0.179193, FRVar 0.058117, and official
TLCC/FRSyn 45.803853. The postprocess changes one frame in each constant AU
of query 0 for 89/1142 eligible contexts; it is an implementation detail, not
a model contribution. The unified six-arm fusion ablation retains raw outputs,
for which TLCC reaches the boundary lag 49 throughout. Its result is in
`published_results/REACT2025_BIDIRECTIONAL_OFFICIAL5_ABLATIONS_20261022.md`;
the older forward-only summaries remain as diagnostics.

Some historical scripts contain provenance paths rooted at
`/home/zhaowenjie/react2025_new` or `/public/zhaowenjie/react2025_new`.
Compatibility links at the old top-level paths are retained, so those scripts
continue to resolve on this machine.
