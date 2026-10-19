#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

EXPECTED = "ff7c642a3e5aa9ab2effb27fc009498464d040321cbce41a3eef8726837b8553"
NAMES = (
    "emotion-query-mamba-epoch0003-seed1.pth",
    "mam_reactor_offline_evidence_epoch0003.pth",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    matches = sorted(
        path
        for name in NAMES
        for path in args.root.resolve().rglob(name)
    )
    if not matches:
        raise SystemExit(f"No P4 candidate found under {args.root.resolve()}")
    exact = []
    for path in matches:
        value = sha256(path)
        mark = "MATCH" if value == EXPECTED else "other"
        print(f"{mark}\t{value}\t{path}")
        if value == EXPECTED:
            exact.append(path)
    if exact:
        print(f"\nCANONICAL_P4={exact[0]}")
    else:
        print("\nNo exact current-main P4 SHA found.")


if __name__ == "__main__":
    main()
