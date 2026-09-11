#!/bin/bash
set -euo pipefail
cd /home/zhengshiyi/react2025_new
.venv/bin/python -m hirp.task_phase25_final --checkpoint runs/phase25/timescale_v1/seed_123/C0/attempt_001/checkpoints/step_004000.pt --device cuda:0 > runs/phase25/timescale_v1/final_seed123_v1/C0_task4000.txt 2>&1
.venv/bin/python -m hirp.task_phase25_final --checkpoint runs/phase25/timescale_v1/seed_123/C0/attempt_001/checkpoints/step_006000.pt --device cuda:0 > runs/phase25/timescale_v1/final_seed123_v1/C0_task6000.txt 2>&1
