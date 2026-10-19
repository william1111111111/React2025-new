"""Train one shared ten-slot student directly from fixed real GT trajectories.

This is the no-distillation control for ``train_expert_student``.  The source
view, model initialization, optimizer, loss and training budget are kept the
same; only the target tensor changes from cached teacher predictions to the
ten fixed real trajectories in a fixed-pair manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_data import (  # noqa: E402
    ConditionalReactionDataset,
    diffusion_listener_crop,
)
from regnn.conditional_model import (  # noqa: E402
    ConditionalREGNNConfig,
    ConditionalREGNNLoss,
)
from regnn.expert_student import (  # noqa: E402
    ExpertConditionalREGNN,
    ExpertStudentConfig,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FixedTenGTDataset(Dataset):
    """One deterministic source view and ten diffusion-compatible real GTs."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        *,
        clip_length: int = 750,
        source_seed: int = 1,
        view_epoch: int = 0,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("split") != "train":
            raise ValueError("Fixed-GT manifest must be TRAIN-only")
        self.mapping = self.manifest["mapping"]
        self.source = ConditionalReactionDataset(
            str(data_root.resolve()), "train", clip_length=clip_length,
            training=True, target_mode="paired", seed=source_seed,
        )
        self.source.set_epoch(view_epoch)
        self.clip_length = int(clip_length)
        missing = [p.as_posix() for p in self.source.records
                   if p.as_posix() not in self.mapping]
        if missing:
            raise ValueError(f"Manifest misses {len(missing)} source records")
        counts = {len(self.mapping[p.as_posix()]) for p in self.source.records}
        if counts != {10}:
            raise ValueError(f"Expected exactly ten fixed GTs, got counts {counts}")

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.source[index]
        record = self.source.records[index]
        start = int(sample["start"])
        targets = []
        for relative in self.mapping[record.as_posix()]:
            raw = self.source._load_tensor(
                self.source.emotion_dir / Path(relative)
            )
            targets.append(diffusion_listener_crop(
                raw, sample["speaker_emotion"], start, self.clip_length,
            ))
        return {
            "speaker_audio": sample["speaker_audio"],
            "speaker_emotion": sample["speaker_emotion"],
            "speaker_3dmm": sample["speaker_3dmm"],
            "length": sample["length"],
            "sample_id": sample["clip_id"],
            "targets": torch.stack(targets),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_commit_sha() -> str:
    configured = os.environ.get("CORESET_GIT_COMMIT")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fixed-gt-manifest", type=Path, required=True)
    parser.add_argument("--initialize-backbone", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--view-epoch", type=int, default=0)
    parser.add_argument("--expert-embedding-dim", type=int, default=64)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data = FixedTenGTDataset(
        args.data_root, args.fixed_gt_manifest,
        source_seed=args.source_seed, view_epoch=args.view_epoch,
    )
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    initialization = torch.load(args.initialize_backbone, map_location="cpu")
    backbone = ConditionalREGNNConfig(**initialization["model_config"])
    config = ExpertStudentConfig(
        backbone=backbone, num_experts=10,
        expert_embedding_dim=args.expert_embedding_dim,
    )
    model = ExpertConditionalREGNN(config).to(device)
    model.backbone.load_state_dict(initialization["state_dict"], strict=True)
    criterion = ConditionalREGNNLoss(ccc_weight=1.0, velocity_weight=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05,
    )

    provenance = {
        "git_commit_sha": source_commit_sha(),
        "method": "direct fixed-ten real-GT student; no teacher outputs",
        "target_alignment": "diffusion_listener_crop (diffusion_legacy)",
        "fixed_gt_manifest_sha256": sha256_file(args.fixed_gt_manifest),
        "initialization_checkpoint_sha256": sha256_file(args.initialize_backbone),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__), MAM_ROOT / "regnn/conditional_data.py",
                MAM_ROOT / "regnn/expert_student.py",
                MAM_ROOT / "regnn/conditional_model.py",
            )
        },
    }
    configuration = {
        **vars(args), "parameter_count": sum(p.numel() for p in model.parameters()),
        "dataset_size": len(data), "num_targets": 10,
        "loss": {"ccc_weight": 1.0, "velocity_weight": 0.05},
        "optimizer": {"name": "AdamW", "weight_decay": 1e-4,
                      "betas": [0.9, 0.99], "grad_clip": 1.0},
    }
    (args.run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (args.run_dir / "config.json").write_text(
        json.dumps(configuration, default=str, indent=2) + "\n"
    )

    started = time.monotonic()
    global_step, epoch_summaries = 0, []
    for epoch in range(args.epochs):
        model.train()
        accumulators: dict[str, float] = {}
        batches = 0
        for batch in loader:
            inputs = [batch[key].to(device, non_blocking=True) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
            )]
            target = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model.generate(*inputs)
                prediction, mask = output["prediction"], output["valid_mask"]
                losses = criterion(
                    prediction.flatten(0, 1), target.flatten(0, 1),
                    mask.flatten(0, 1),
                )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Nonfinite loss at step {global_step}")
            losses["loss"].backward()
            gradient_norm = clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            batches += 1
            for key, value in losses.items():
                accumulators[key] = accumulators.get(key, 0.0) + float(value.detach())
            accumulators["gradient_norm"] = (
                accumulators.get("gradient_norm", 0.0) + float(gradient_norm)
            )
            if global_step % 100 == 0:
                print(json.dumps({
                    "epoch": epoch + 1, "step": global_step,
                    **{key: value / batches for key, value in accumulators.items()},
                }, sort_keys=True), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        scheduler.step()
        summary = {
            "epoch": epoch + 1, "step": global_step,
            **{key: value / batches for key, value in accumulators.items()},
        }
        epoch_summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break

    checkpoint = {
        "state_dict": model.state_dict(),
        "student_config": {
            "backbone": asdict(config.backbone),
            "num_experts": config.num_experts,
            "expert_embedding_dim": config.expert_embedding_dim,
        },
        "global_step": global_step,
        "supervision": "fixed_real_gt_no_distillation",
        "fixed_gt_manifest_sha256": provenance["fixed_gt_manifest_sha256"],
        "initialization_checkpoint_hash": provenance["initialization_checkpoint_sha256"],
        "provenance": provenance,
    }
    temporary = args.run_dir / "student.pth.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.run_dir / "student.pth")
    (args.run_dir / "train_metrics.json").write_text(
        json.dumps(epoch_summaries, indent=2) + "\n"
    )
    configuration.update({
        "global_steps": global_step,
        "training_seconds": time.monotonic() - started,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    })
    (args.run_dir / "config.json").write_text(
        json.dumps(configuration, default=str, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
