# Observable mode supervision v1

New independent T0-14000 experiment on all 1,660 TRAIN sources. The BERT run is complete and is not its parent. No API calls, human labels, event classes, shared local noise, candidate identities, or transport assignments are used.

`prepare.py` audits actual source video PTS and feature lengths, retaining raw paired time axes. One paired target is one frame shorter; its true target length is preserved. `plans.py` uses fixed-sign QR of a constant and three PTS cosine bases; incomplete blocks have explicit masks. TRAIN paired recordings fit the fixed coordinate statistics. Same-session alternatives retain the historical full-length interpolation and fixed slot selection, not a new claim of paired timing.

`cache.py` freezes speaker-only condition summaries (all 1,660 recordings). `models.py` implements a four-layer width-256 recording-plan flow and zero-final-projection residual injection. `train.py` separates the prior optimizer from decoder FM/task/mode losses. Each formal arm inherits the same T0 weights and optimizer, trains 6,000 decoder steps, and uses the same source/target/block/local-noise schedule. M1 has 2,000 prior warmup steps and 6,000 concurrent prior updates. Its one-time TRAIN calibration precedes formal decoder updates; the mode weight ramps for 200 steps. M0 does not use the prior or injection. Added compute is explicitly unmatched.

Run with `CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m mode_supervision.queue` after preparation, caching, acceptance, and identity freezing. The finite queue runs both arms on card 7, evaluates checkpoints 1,000/3,000/6,000 on DEV80, and runs exact FRD20 at the 6,000 endpoint. Fixed-plan, fixed-noise, and reference-plan diagnostics are separate. No automatic push or budget extension.

Checks: `test_plans.py` (8 tests), `acceptance.py` (real TRAIN exposure/plan correspondence, zero initialization, gradients and K-prefix), and `check_resume.py` (continuous three steps versus two plus resume, including both optimizers and RNG). Smoke checkpoints cannot be used for formal evaluation.

Artifacts: `runs/reaction_flow/mode_supervision_v1/`. `PROTOCOL.json` fixes the strict quality criteria before training; `cleanup.json` records removed old test checkpoints. No new formal task-performance claims exist before evaluation completes. Sensor-level synchronization is not independently verified merely by matching feature/video frame counts.
