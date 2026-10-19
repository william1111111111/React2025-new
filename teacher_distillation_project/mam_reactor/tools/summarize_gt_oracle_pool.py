#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


P4_SHA256 = "ff7c642a3e5aa9ab2effb27fc009498464d040321cbce41a3eef8726837b8553"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    examples = payload["examples"]

    variant_counts = collections.Counter()
    slot_counts = collections.Counter()
    swap_counts = collections.Counter()
    gains = []
    rows = []

    for example in examples:
        selected = example["selected_ids"]
        for item in selected:
            variant, slot = item.split("/q")
            if variant != "baseline":
                variant_counts[variant] += 1
                slot_counts[int(slot)] += 1
        for event in example.get("swaps", []):
            swap_counts[event["kind"]] += 1
        gain = float(example["selected_frdiv"] - example["baseline_frdiv"])
        gains.append(gain)
        rows.append(
            (
                example["clip_id"],
                example["sequence_length"],
                example["pool_size"],
                gain,
                example["selected_frc"] - example["baseline_frc"],
                example["selected_frd"] - example["baseline_frd"],
            )
        )

    lines = []
    lines.append("# Mam-Reactor GT-Oracle Candidate-Pool Results")
    lines.append("")
    checkpoint = Path(payload["checkpoint"])
    checkpoint_sha256 = file_sha256(checkpoint)
    lines.append("## Scope and provenance")
    lines.append("")
    lines.append(f"- Checkpoint: `{checkpoint}`")
    lines.append(f"- Checkpoint SHA-256: `{checkpoint_sha256}`")
    lines.append(
        "- Current-main P4 identity: "
        + ("MATCH" if checkpoint_sha256 == P4_SHA256 else "NOT MATCHED")
    )
    lines.append(
        f"- Split: `{payload['split_name']}`; contexts: "
        f"`{summary['num_examples']}`; seed: `{payload['selection_seed']}`"
    )
    lines.append(f"- Scope: {payload['scope']}")
    lines.append(
        "- GT construction: train-side B_fit files only; ten same-session "
        "listener GTs selected with the official test-style rule and aligned "
        "to the full speaker sequence by the official target Processor."
    )
    lines.append(
        "- These train-side pilot scores are not the archived 1,142-sample "
        "full-test scores and are not directly comparable to them."
    )
    lines.append("")
    lines.append("## Main result")
    lines.append("")
    b = summary["baseline"]
    o = summary["oracle_selected"]
    d = summary["delta"]
    lines.append(
        f"- FRC: `{b['FRC']:.6f}` → `{o['FRC']:.6f}` "
        f"(Δ `{d['FRC']:+.6f}`)"
    )
    lines.append(
        f"- exact FRD: `{b['exact_FRD']:.6f}` → `{o['exact_FRD']:.6f}` "
        f"(Δ `{d['exact_FRD']:+.6f}`)"
    )
    lines.append(
        f"- FRDiv: `{b['FRDiv']:.6f}` → `{o['FRDiv']:.6f}` "
        f"(Δ `{d['FRDiv']:+.6f}`)"
    )
    lines.append(
        f"- Positive-FRDiv contexts: "
        f"`{summary['examples_with_positive_frdiv_gain']}/{summary['num_examples']}` "
        f"(`{summary['positive_frdiv_gain_fraction']:.1%}`)"
    )
    lines.append(
        f"- Median / max FRDiv gain: "
        f"`{summary['median_frdiv_gain']:+.6f}` / "
        f"`{summary['max_frdiv_gain']:+.6f}`"
    )
    lines.append("")
    lines.append("## Selected non-baseline variants")
    lines.append("")
    if variant_counts:
        for name, count in variant_counts.most_common():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No non-baseline variants were selected.")
    lines.append("")
    lines.append("## Slot replacement counts")
    lines.append("")
    if slot_counts:
        for slot, count in sorted(slot_counts.items()):
            lines.append(f"- q{slot}: {count}")
    else:
        lines.append("- No slots were replaced.")
    lines.append("")
    lines.append("## Search operations")
    lines.append("")
    lines.append(f"- one-swap: {swap_counts['one_swap']}")
    lines.append(f"- two-swap: {swap_counts['two_swap']}")
    lines.append("")
    lines.append("## Per-context oracle headroom")
    lines.append("")
    lines.append("| clip | T | pool | ΔFRDiv | ΔFRC | ΔFRD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for clip, length, pool, gain, frc, frd in sorted(rows, key=lambda x: x[3], reverse=True):
        lines.append(
            f"| `{clip}` | {length} | {pool} | {gain:+.6f} | "
            f"{frc:+.6f} | {frd:+.6f} |"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
