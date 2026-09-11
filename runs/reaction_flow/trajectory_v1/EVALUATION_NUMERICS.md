# Fixed evaluation backend, math_v2

The first A2000 evaluation stopped before a complete score; its first export and log are retained.
Production inference now explicitly uses torch SDPA math, matching training and GPU regression;
FP32, Euler16, noise, checkpoints, official metrics and target population are unchanged.
The same setting applies to all checkpoint evaluations, not selected clips.

On the failing source, with identical noise under explicit math: FP32 K10/chunk3 max difference
1.0848045349121094e-5; FP64 max1.6042722705833512e-14; FP32-to-FP64 max
1.6854902682439388e-5 and mean2.985282418623148e-8. No target quality scores were
used to choose this backend. The pre-existing float64 fallback tolerances are unchanged.

New prediction cache namespace: exports/math_v2. No historic cache metadata was rewritten.
The training process continued without interruption. Only controller1457878 was paused;
queue --adopt tracks the same training status and owns subsequent evaluations.
