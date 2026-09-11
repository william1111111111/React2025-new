# Actual commands and versions

Workspace HEAD83f79700cb3569fc7b2fc7288e39b02c6f92594c; branch agent/r2-capacity-cover-refinement.
Uncommitted user/experiment files retained. No push, reset, stash, clean or old-data deletion.

- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest mam_staged/tests mam_target/tests -q`
  Logs: staged_v1/tests.txt, tests_v2.txt, tests_v3.txt. Latest completed:11passed.
- `CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 .venv/bin/python -m mam_staged.calibrate`
  Original/retry1/retry2 logs retained in staged_v1. Retry3 completed TRAIN16 calibration.
  Its calibration JSON is reused byte-for-byte; no DEV-based recalibration.
- `CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 .venv/bin/python -m mam_staged.regression`
  Completed staged_v1; repeated staged_v2 after initial guard correction. Per-attempt status and GPU regression JSON specify actual results.
- `python -m mam_staged.queue` launched in tmux on GPU1 for staged_v1; interrupted as a numerical trial
  (all three arms retained as incomplete, not used for method results).
- Staged_v2 formal commands/PIDs/exit codes are recorded individually by the queue in `*_command.json` / `*_exit.json`.

The earlier A/B PIDs1838933/1838934/1838935 were verified stopped (T) and were never resumed.
New numerical-trial scheduler2469192 was SIGSTOP'd; only its training children2470992/2470993/2471010
received SIGINT. Their incomplete histories remain in staged_v1.
