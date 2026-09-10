# Phase 2.4 executed commands

Workspace `/home/zhengshiyi/react2025_new`; clean initial HEAD `8d7e788a88b494882bed1f5565807796da5fb071`, no later commits, no AGENTS.md in workspace/ancestors. `.agents` and `.codex` were empty. Created local branch `agent/hirp-phase24-multitarget-evaluation`; no push.

Resource check: `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`. GPU6 A100 80GB was empty (4 MiB) at launch; other occupied GPUs were left alone. This round's MC and task evaluation share GPU6. Wall times include that contention and CPU scoring, not isolated GPU benchmarks.

Executed preparation/regression:

```bash
.venv/bin/python -m pytest hirp/tests/test_phase24.py hirp/tests/test_phase23.py -q > runs/phase24/evaluation_v1/tests_cache.txt 2>&1
.venv/bin/python -m hirp.setup_phase24 > runs/phase24/evaluation_v1/setup_stdout.txt 2>&1
.venv/bin/python -m pytest hirp/tests -q > runs/phase24/evaluation_v1/tests_full.txt 2>&1
.venv/bin/python -m pytest hirp/tests/test_phase24.py -q > runs/phase24/evaluation_v1/tests_selection.txt 2>&1
```

Results: 13 passed initial targeted tests; 124 passed/3 skipped/1 xfailed full regression; 5 passed after adding the literal-official-AST target-selection comparison. Existing CPU autocast limitation remains xfailed; real GPU processor/exports are exercised separately. Setup froze 15 candidates, exactly 8 new banks and 80 sources before any new scores were read.

Executed fixed evaluations:

```bash
.venv/bin/python -m hirp.evaluate_phase24 --device cuda:6 > runs/phase24/evaluation_v1/mc_stdout.txt 2>&1
.venv/bin/python -m hirp.task_phase24 --device cuda:6 > runs/phase24/evaluation_v1/task_stdout.txt 2>&1
.venv/bin/python -m hirp.task_phase24 --device cuda:6 --mam-only > runs/phase24/evaluation_v1/mam_stdout.txt 2>&1
```

MC has completed all120 cases. MAM has completed all80 source exports and fixed multi-target/single-paired metrics. Each HiRP candidate is evaluated in the same fixed manifest order; exact completion is recorded in `task_multitarget/completed_HiRP.json`. No training command was run.

Read-only provenance inspection covered the actual loader, Processor, FRC/FRVar/S-MSE/TLCC/FRD implementations and `mam_reactor` README/PROVENANCE/infer.sh/manifest/config. The archive's documented original `/tmp/react2025-joint-va-main-ablation-20260817` directory no longer exists, so `git -C ... rev-parse HEAD` could not recover the original source commit. Immutable archive hashes, native inference settings and the representative weight were verified instead. No external historical metric was copied into the new task tables.

`unequal_processor_tests.json` records the actual 958→1926, 1926→1167, 2253→1053 cases; same-RNG prediction-value invariance error0. Normalization alignment with MAM was checked by file hashes. `unchanged_metric_regression.json` records zero aggregate difference from the old fixed noise/input case.

All new score/candidate/manifest artifacts stay under `runs/phase24/evaluation_v1`. Arrays/noise banks/processed targets/full exports are local and gitignored; checkpoint files are only read at existing paths. No credentials or environment files are added.

Final aggregation executed after all15 HiRP task models completed:

```bash
.venv/bin/python -m hirp.report_phase24 > runs/phase24/evaluation_v1/report_stdout.txt 2>&1
.venv/bin/python -m hirp.write_phase24_report > runs/phase24/evaluation_v1/write_report_stdout.txt 2>&1
.venv/bin/python -m hirp.write_phase24_report > runs/phase24/evaluation_v1/write_report_final_stdout.txt 2>&1
```

The report's final opening explicitly distinguishes the observed six positive C2-C1 multi-target FRC differences, the paired-only direction differences, and MAM's higher task score. No measurement was changed by that prose revision. Source dependency check after all runs remained identical to the frozen snapshot. MC wall time was565.401s on GPU6 with this round's task workload sharing the device. Per-model task wall/peak-memory records are in each result JSON.

Final read-only verification and strengthened temporary-copy regression:

```bash
.venv/bin/python -m hirp.verify_phase24 > runs/phase24/evaluation_v1/verify_stdout.txt 2>&1
.venv/bin/python -m pytest hirp/tests/test_phase24.py -q > runs/phase24/evaluation_v1/tests_dependency_final.txt 2>&1
.venv/bin/python -m hirp.write_phase24_report > runs/phase24/evaluation_v1/write_report_delivery_stdout.txt 2>&1
```

Verification passed: all120 MC cases, 15 HiRP task cases, one MAM case, nine old single-paired metric regressions, and2151 Phase23 artifacts unchanged. The last focused regression uses actual generator/population source copies in temporary directories and an existing cache behind the normalization guard; all5 tests passed. No production files were mutated by these tests. Plots were visually inspected. Final artifact index is refreshed only after logs/report finish writing, avoiding self-referential output hashes.
