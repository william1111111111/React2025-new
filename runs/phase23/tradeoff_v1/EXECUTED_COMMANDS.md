# Executed Phase 2.3 commands

Working directory: `/home/zhengshiyi/react2025_new`. Baseline `135220fae7f9018eec8f8119a72b4e2d074f713b`; initial working tree clean. Created `agent/hirp-phase23-tradeoff-evaluation`. No push.

## Preparation and regression

- `.venv/bin/python -m hirp.setup_phase23 > runs/phase23/tradeoff_v1/setup_stdout.txt 2>&1` — initial preparation stopped on tuple-vs-JSON-list config equality. No training was executed by this attempt. The log is preserved.
- Same setup command with `setup_retry_stdout.txt` after canonical config comparison — succeeded; existing initial audit preserved.
- `.venv/bin/python -m pytest hirp/tests/test_phase23.py -q > runs/phase23/tradeoff_v1/tests_initial.txt 2>&1` — 8 passed.
- `.venv/bin/python -m pytest hirp/tests/test_phase23.py hirp/tests/test_phase22.py -q > runs/phase23/tradeoff_v1/tests_cache_contract.txt 2>&1` — 22 passed, including actual noise/source/scaler content mutation and complete-cache no-op.
- `.venv/bin/python -m pytest hirp/tests -q > runs/phase23/tradeoff_v1/tests_full.txt 2>&1` — 120 passed, 3 skipped, 1 xfailed. No old training matrix rerun.
- `.venv/bin/python -m hirp.audit_phase23_long --device cuda:7 > runs/phase23/tradeoff_v1/long_audit_stdout.txt 2>&1` — actual trained model, T128/256/750 and full 1001-frame reassembly, completed.
- `.venv/bin/python -m hirp.task_phase23 --device cuda:7 > runs/phase23/tradeoff_v1/task_stdout.txt 2>&1` — all nine reused models, all Development-80 full source sequences, K10, unchanged local metric implementations; completed.
- `.venv/bin/python -m hirp.visualize_phase23 > runs/phase23/tradeoff_v1/visualize_stdout.txt 2>&1` — four pre-fixed full-trajectory cases, all 25 channels and 10 samples.

## Finite training matrix

Each invocation runs exactly C1/C2 × lambda .03/.3, fresh initialization and 2000 actual optimizer steps per arm. Seed123 completed before the other two launches.

```bash
.venv/bin/python -m hirp.train_phase23 --seed 123 --device cuda:4 > runs/phase23/tradeoff_v1/train_seed123_stdout.txt 2>&1
.venv/bin/python -m hirp.train_phase23 --seed 42 --device cuda:4 > runs/phase23/tradeoff_v1/train_seed42_stdout.txt 2>&1
.venv/bin/python -m hirp.train_phase23 --seed 2026 --device cuda:6 > runs/phase23/tradeoff_v1/train_seed2026_stdout.txt 2>&1
```

## Final evaluation and learning curves

The first three commands evaluate only the reused points; subsequent full-frontier invocations validate and reuse those exact complete identities.

```bash
.venv/bin/python -m hirp.evaluate_phase23 --seed 123 --device cuda:6 --only-reused > runs/phase23/tradeoff_v1/eval_reused123_stdout.txt 2>&1
.venv/bin/python -m hirp.evaluate_phase23 --seed 42 --device cuda:7 --only-reused > runs/phase23/tradeoff_v1/eval_reused42_stdout.txt 2>&1
.venv/bin/python -m hirp.evaluate_phase23 --seed 2026 --device cuda:6 --only-reused > runs/phase23/tradeoff_v1/eval_reused2026_stdout.txt 2>&1
.venv/bin/python -m hirp.evaluate_phase23 --seed 123 --device cuda:7 > runs/phase23/tradeoff_v1/eval_seed123_stdout.txt 2>&1
.venv/bin/python -m hirp.evaluate_phase23 --seed 42 --device cuda:4 > runs/phase23/tradeoff_v1/eval_seed42_stdout.txt 2>&1
.venv/bin/python -m hirp.evaluate_phase23 --seed 123 --device cpu > runs/phase23/tradeoff_v1/cache_repeat_seed123_stdout.txt 2>&1
.venv/bin/python -m hirp.curve_phase23 --seed 123 --device cuda:7 > runs/phase23/tradeoff_v1/curve_seed123_stdout.txt 2>&1
.venv/bin/python -m hirp.curve_phase23 --seed 42 --device cuda:7 > runs/phase23/tradeoff_v1/curve_seed42_stdout.txt 2>&1
```

`cache_repeat_seed123_stdout.txt` records 28 `REUSED` rows and `COMPLETE reused`: no metric calculation and no completion-file collision. A changed identity is an error; an intentional full recomputation uses a new `--evaluation-name` directory. Historical result files have not been given fabricated new identity labels.

Code compilation and read-only git/status/hash inspection were also executed. Runtime source snapshots and result/checkpoint fingerprints are retained alongside the logs. Later commands and final completion states are appended after execution.

Executed final-seed commands:

```bash
.venv/bin/python -m hirp.evaluate_phase23 --seed 2026 --device cuda:6 > runs/phase23/tradeoff_v1/eval_seed2026_stdout.txt 2>&1
.venv/bin/python -m hirp.curve_phase23 --seed 2026 --device cuda:4 > runs/phase23/tradeoff_v1/curve_seed2026_stdout.txt 2>&1
```

All 12 new training summaries now report `completed=true`, `requested_steps=actual_optimizer_steps=training_rows=2000`; all starts were from random initialization. New total: 24,000 optimizer steps, 96,000 source occurrences. The nine old trainings were reused, not retrained.

Final aggregation and verification executed:

```bash
.venv/bin/python -m hirp.report_phase23 > runs/phase23/tradeoff_v1/report_stdout.txt 2>&1
.venv/bin/python -m hirp.verify_phase23 > runs/phase23/tradeoff_v1/verify_stdout.txt 2>&1
.venv/bin/python -m hirp.write_phase23_report > runs/phase23/tradeoff_v1/write_report_stdout.txt 2>&1
.venv/bin/python -m hirp.write_phase23_report > runs/phase23/tradeoff_v1/write_report_final_stdout.txt 2>&1
```

The final prose revision explicitly added the observed K32/K64 conditional-screen instability; no training/evaluation was repeated or selected away. Final result counts: 84 final bank/K cases, 105 checkpoint learning-curve cases, 9 full-source task evaluations. Historical preservation check: 624 files, zero changed. Phase22 numerical regression: 36 cases, maximum aggregate difference 0.

CPU plotting emitted a matplotlib configuration-directory warning and used its `/tmp` cache successfully. No results were suppressed because of a warning. An explicit finite-support algebra calculation saved `algebra_identity.json`; its absolute residual is 8.326672684688674e-17. All runtime generator/scoring sources were copied to `source_snapshots/` before final reporting.
