"""Validate minimal annotation sidecars and prevent obvious train/test role leakage.

Requires jsonschema. This checks metadata and time fields, not semantic truth or
whether an annotator actually followed the access contract. Real runs additionally
need media hashes, controlled loading and audit logs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("annotation_minimal.schema.json")

def validate(record: dict[str, Any], *, allow_examples: bool = False) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)
    if record["is_synthetic_example"] and not allow_examples:
        raise ValueError("Synthetic examples must never enter a training dataset")
    kind = record["record_type"]
    roles = set(record["visible_roles"])
    if kind == "speaker_events":
        if roles != {"speaker"} or not record["allowed_for_model_input"]:
            raise ValueError("Source annotations must be speaker-only")
    else:
        if record["split"] != "train" or record["allowed_for_model_input"]:
            raise ValueError("Listener/relationship supervision is TRAIN-only, never inference input")
        expected = {"listener"} if kind == "listener_observations" else {"speaker", "listener"}
        if roles != expected:
            raise ValueError("Annotation role provenance does not match its type")
    seen: set[str] = set()
    for event in record["events"]:
        if not 0 <= event["start_s"] <= event["end_s"] <= record["duration_s"]:
            raise ValueError("Invalid event interval in absolute clip time")
        if event["event_id"] in seen:
            raise ValueError("Duplicate event identifier")
        seen.add(event["event_id"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--allow-examples", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]
    for i, record in enumerate(records):
        try:
            validate(record, allow_examples=args.allow_examples)
        except Exception as exc:
            raise SystemExit(f"Record {i} rejected: {exc}") from exc
    print(f"Validated {len(records)} metadata records; semantic correctness not assessed.")
