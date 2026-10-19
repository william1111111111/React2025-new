#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

EXPECTED_P4_SHA256 = "ff7c642a3e5aa9ab2effb27fc009498464d040321cbce41a3eef8726837b8553"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    required = [
        "dataset/react_2025.py",
        "regnn/emotion_query_mamba.py",
        "regnn/query_mamba.py",
        "regnn/conditional_model.py",
        "regnn/eval_conditional_regnn_official_test.py",
        "framework/metrics/FRC.py",
        "framework/metrics/FRD.py",
        "framework/metrics/S_MSE.py",
        "tools/compute_frd_resumable.py",
        "regnn/probe_gt_oracle_candidate_pool.py",
        "regnn/configs/eqr_session_splits_seed1.json",
    ]
    missing = [item for item in required if not (root / item).is_file()]
    if missing:
        raise SystemExit("Missing required repo files:\n  " + "\n  ".join(missing))

    import torch
    from regnn.emotion_query_mamba import EmotionQueryResidualMamba
    from regnn.query_mamba import apply_residual_channel_multipliers
    from tools.compute_frd_resumable import rolling_dtw

    signature = inspect.signature(EmotionQueryResidualMamba.forward)
    required_parameters = {
        "residual_scale",
        "style_residual_scale",
        "emotion_query_gates_override",
        "mixture_weights_override",
    }
    missing_parameters = sorted(required_parameters - set(signature.parameters))
    if missing_parameters:
        raise SystemExit(
            "Local Mam-Reactor API is too old; missing forward args: "
            + ", ".join(missing_parameters)
        )

    if not callable(apply_residual_channel_multipliers):
        raise SystemExit("apply_residual_channel_multipliers is unavailable")
    if not callable(rolling_dtw):
        raise SystemExit("rolling_dtw is unavailable")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; generation cannot run.")

    if args.data_dir is not None:
        data = args.data_dir.resolve()
        data_required = [
            data / "train" / "facial-attributes",
            data / "train" / "audio-features",
            data / "train" / "coefficients",
            data / "train" / "video-face-crop",
        ]
        absent = [str(path) for path in data_required if not path.exists()]
        if absent:
            raise SystemExit("Dataset paths missing:\n  " + "\n  ".join(absent))
        print(f"DATA OK: {data}")

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        actual = sha256(checkpoint)
        print(f"Checkpoint SHA256: {actual}")
        if actual != EXPECTED_P4_SHA256:
            print(
                "WARNING: checkpoint SHA differs from the archived current-main P4 "
                "checkpoint. Do not interpret results as the P4 probe "
                "until provenance is resolved."
            )
        else:
            print("Canonical current-main P4 checkpoint: MATCH")
        payload = torch.load(checkpoint, map_location="cpu")
        print(f"Checkpoint method: {payload.get('method')}")
        if payload.get("method") not in {"EQR-Mamba", "Emotion-Gated-Query-Mamba"}:
            raise SystemExit("Checkpoint is not an EQR/Mam-Reactor checkpoint.")

    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
