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
50% overlap, center-weighted full-sequence inference. On forward VAL571 it
improves the old full non-overlap baseline from FRC 1.120550 / exact FRD
143.006997 to FRC 1.223777 / exact FRD 135.472731, while FRDiv and FRVar fall
to 0.167050 and 0.058290. This is a measured quality/continuity versus
diversity trade-off; see
`published_results/REACT2025_MULTIVIEW_BOUNDARY_STUDENT_20261018.md`.

A minimal follow-up sharpens the triangular fusion weights from `w` to
normalized `w^4`. It recovers 60.24% of the overlap-induced FRDiv gap while
retaining 85.99% of the FRC gain, without changing the network or training;
see `published_results/REACT2025_POWERED_OVERLAP_20261018.md`.

Some historical scripts contain provenance paths rooted at
`/home/zhaowenjie/react2025_new` or `/public/zhaowenjie/react2025_new`.
Compatibility links at the old top-level paths are retained, so those scripts
continue to resolve on this machine.
