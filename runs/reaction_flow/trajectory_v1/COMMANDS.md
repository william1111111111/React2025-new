# Commands actually executed

- `git switch -c agent/reaction-flow-conditional-law` from211383b03b388293bf2b231f6a9b3d66052ff9c7, preserving uncommitted final staged results.
- `OMP_NUM_THREADS=1 .venv/bin/python -m reaction_flow.prepare` — all recorded TRAIN/normalization hashes verified;14000-record schedule and TRAIN coordinate stats generated.
- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest reaction_flow/tests -q` — latest tests_final.txt:5passed.
- GPU1: `python -m reaction_flow.train --stop 2 --label smoke_continuous` — two real updates, no completed12000-step claim.
- GPU1: `python -m reaction_flow.regression` — real2vs1+1 resume, K10 chunk and TRAIN Euler16/32 probe.
- GPU1: `python -m reaction_flow.task_probe` — pure-noise differentiable Euler16/K4 task gradient; zero optimizer updates.
- `tmux new-session -d -s reaction_flow_A_v1 ... python -m reaction_flow.queue` — real background phase-A run and fixed checkpoints/evaluation.

GPU commands use CUDA_VISIBLE_DEVICES=1,CUBLAS_WORKSPACE_CONFIG=:4096:8,OMP_NUM_THREADS=1.
Actual child command arguments/PIDs/exit codes are stored by queue in *_command.json/*_exit.json.
No other seeds, other-user process signals, push, old-asset deletion, or model weights publication.
F-cont/F-task implementation exists; its calibration and training have not run, pending A12000 inspection.
Evaluation noise hashes are calculated from actual per-recording tensors before any cache reuse;
this evaluator hardening was completed before the first evaluation, without altering training code or budget.
