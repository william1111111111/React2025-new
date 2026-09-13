# Evidence-anchored semantic supervision pilot

Task specification: `CODEX_SEMANTIC_SUPERVISION.md`. Original supplied pack is preserved in `vendor/react_semantic_supervision/`, with attachment hashes. Its synthetic examples are never mixed into actual TRAIN annotations.

This first delivery performs CPU media/PTS and numeric proposal extraction plus local human-review export. It does not implement/launch a semantic generator, invoke ASR/LLM APIs, or claim reviewed semantic labels.

From the repository root:

```bash
.venv/bin/python -m unittest discover -s semantic_supervision/vendor/react_semantic_supervision -p test_validation.py -v
.venv/bin/python -m pytest semantic_supervision/tests -q
.venv/bin/python -m semantic_supervision.extract.run_pilot
.venv/bin/python -m semantic_supervision.review.export
```

The extraction launcher uses Linux bubblewrap mount/PID/network namespaces, clean environment, read-only runtime/code/media mounts, and writable stage-specific outputs. Source stage mounts only TRAIN speaker WAV/video/TXT; listener stage mounts only TRAIN listener video/attributes; relation stage mounts only the two TRAIN output folders. No host project/data root, credentials, network or NVIDIA device nodes are mounted. Native subprocesses inherit that isolation. Runtime imports execute from a read-only local virtualenv. Isolation fails closed if bubblewrap cannot start. Re-running requires a new/preserved output attempt because stage execution logs use exclusive creation.

Outputs: `runs/reaction_flow/semantic_supervision_pilot_v1/`. Individual frame PTS are measured with ffprobe, not inferred from a global FPS assumption. Audio windows use the actual sample clock. Unknown audio/video offsets and channel-role ambiguity prevent acceptance. Listener200ms smoothing follows measured PTS; the90th-percentile numeric-change detector is fit only on the34 TRAIN pilot recordings. It is a candidate detector, not a visually verified observation labeler.

No new checkpoints or official metrics are written. E0/E-event and necessary E-text control settings are draft in `models/experiment_config.json`; budget and annotation acceptance policy remain unfilled until real review. E-plan and event retiming are deferred and must not be activated together with the first event-input experiment.
