#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${DATA_DIR:?Set DATA_DIR to the REACT2025 dataset root}"
: "${CHECKPOINT:?Set CHECKPOINT to the selected Mam-Reactor Full checkpoint}"

GPU_ID="${GPU_ID:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-24}"
SCORE_WORKERS="${SCORE_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-bfit_strict_n${MAX_EXAMPLES}_seed1234}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/mam_reactor_gt_oracle_pool/${RUN_TAG}.json}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" regnn/probe_gt_oracle_candidate_pool.py \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --session-split-manifest regnn/configs/eqr_session_splits_seed1.json \
    --split-name B_fit \
    --output-json "${OUTPUT_JSON}" \
    --max-examples "${MAX_EXAMPLES}" \
    --selection-seed 1234 \
    --score-workers "${SCORE_WORKERS}" \
    --frc-slack 0 \
    --frd-slack 0 \
    --include-mixture-sweep

echo "RESULT_JSON=${OUTPUT_JSON}"
