# Reaction Program annotation preparation v1

This directory contains **proposals**, not completed action labels. TRAIN/VAL/TEST caches are physically separate. `train/speaker_events.jsonl` contains exact source transcript evidence. `listener_reactions.jsonl` contains pending real-media observation tasks; `actions=null` means unknown. Equivalence and support candidates make no positive match claims. No paid request/media upload is implemented or executed by this preparation.

`train/requests/*.jsonl` are allowlisted annotator inputs, with opaque asset IDs. `ASSET_RESOLVER_PRIVATE.json` and `RECORDING_MAP_PRIVATE.json` are execution/provenance sidecars, NEVER model/teacher prompt text. A dispatcher must resolve only each task's declared roles, attach actual media, record accessed modalities, and refuse inaccessible evidence. A text-only model cannot validate facial action labels. Session only chooses potential cross-recording pairs; the equivalence prompt does not receive it.

`train/pts/` preserves existing measured per-frame timestamps. Matching row counts is not independent proof of feature/video synchronization. Most new candidates lack resolved WHO/WHEN. Never interpret a whole-record media request as a precisely localized event window.

Raw annotation records must be append-only: raw response, request hash, actual returned model/version (explicitly unknown if provider does not supply a revision), prompt hash, input hashes, evidence and validation result. The currently supplied DeepSeek responses are historical source-text proposals only; they are not listener visual labels. `confidence=null` means not provided, and self-confidence is never a calibrated probability.

Current discovery subset is TRAIN RM_fit only using conservative recording-date groups. RM_cal and reused RM_audit are not ontology-fit data. These are not certified person/dyad-disjoint folds. Source paths/dates/session IDs cannot enter later program/planner features. A future program importer must validate observation/support foreign keys and prevent cross-fold donor leakage before exposing any loss.

Reproduce from repository root: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m reaction_program.preparation.build`. Exclusive output creation refuses overwrite; use a new version for a rerun. Validation: `.venv/bin/python -m reaction_program.preparation.validate`. There is no codec, executor or planner training in this step.
