# Executed in this session

Workspace audit: `git status --short`, `git branch --show-current`, `git rev-parse HEAD`; baseline7e6adf5, initially clean. `git switch -c agent/hirp-phase25-timescale-alignment`. No push.
Read local source and `/home/zhengshiyi/.codex/skills/run-experiment/SKILL.md`; no AGENTS found in workspace or parents.
`nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`: sandbox driver access failed; escalated read succeeded. All8 cards busy, GPU0 about18GB/80GB,59% utilization on second read. Shared-GPU choice requested, not received as of preparation.

- `.venv/bin/python -m hirp.diagnose_phase25`
- `.venv/bin/python -m pytest hirp/tests/test_phase25.py -q -s > runs/phase25/timescale_v1/tests_temporal.txt 2>&1` (first6 tests)
- `.venv/bin/python -m pytest hirp/tests/test_phase25.py -q -s > runs/phase25/timescale_v1/tests_resume.txt 2>&1` (9 tests after resume tests added)
- `.venv/bin/python -m pytest hirp/tests -q > runs/phase25/timescale_v1/tests_full.txt 2>&1` (134passed,3skipped,1xfailed; CPU)
- Python `hirp.train_phase25.prepare` for seeds123/42/2026, TrainConfig defaults, into `seed_<seed>`: complete6000-step schedules, fresh noise, shared new scaler. `prepare_stdout.txt`.
- Python `RealData.batch(records[0])` twice for each seed, tensor-exact repeated crop/descriptor assertions: `data_contract_check.txt` and JSON.
- `PYTHONPATH=. .venv/bin/python /tmp/audit_phase25.py > runs/phase25/timescale_v1/schedule_audit.txt 2>&1`; exact saved script `audit_schedule_command.py`.
- Python `train(..., device='cpu', stop_after=1)` for C0/C1/C2, actual full model/realT750/B4/K4, full6000-step schedules; artifacts `cpu_smoke`, log `cpu_real_smoke.txt`. Separate smoke directories, excluded from formal results.
- Python C2 real-data `train(stop_after=2)` vs `train(resume=cpu_smoke/C2/.../step_000001.pt,stop_after=2)`, assert model and optimizer exact and training_rows identical; `cpu_real_resume.txt`, `real_resume_regression.json`.
- `.venv/bin/python -m hirp.frd_phase25 --max-pairs 1 > runs/phase25/timescale_v1/frd_smoke.txt 2>&1`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m hirp.frd_phase25 > runs/phase25/timescale_v1/frd_mam.txt 2>&1` (MAM20 full-frame subset, bounded7200seconds/2000pairs, resumes first pair).
- Python exact FRD `_func` vs pairwise weighted DTW on2candidates×3targets, unequal17/21frames; absolute error0, `frd_implementation_regression.json`.
- Python recompute Phase24 `artifact_fingerprints.json`:2833checked,0mismatch; `historical_preservation.json`.

# Prepared but not executed

GPU smoke, GPU-specific regression, and formal training/evaluation require a free card or approved sharing. First segment command after resource preflight:
`.venv/bin/python -m hirp.run_phase25 --device cuda:0 --seed 123 --through 2000 --evaluate`
This uses new seed123/C0,C1,C2 arm directories (CPU smoke is separate), writes command and source snapshots, resumes only explicit durable checkpoints, evaluates fixed128/500/1000/2000 steps. Do not run until GPU memory smoke is completed.
Continuation uses same runner `--through 6000`, then seeds42/2026, identical frozen budget. No training-based selection is implemented.

Final additional checks: `.venv/bin/python -m pytest hirp/tests/test_phase25.py -q > runs/phase25/timescale_v1/tests_final_temporal.txt 2>&1`:10passed, including padded-long tensor with only128 valid frames. `python -m py_compile` for new executable modules passed; `git diff --check` passed. Coverage plot generated deterministically from saved C2_minus_C1.csv, no new model output.
