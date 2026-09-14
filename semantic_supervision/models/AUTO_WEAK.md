# Automatic weak supervision interface

`source_cache.py` remains the gold contract. `auto_weak.py` is an explicit separate reader: callers provide the cache digest, actual speaker-media hashes, policy hash and split. Rejected or missing labels return `None`, not a no-event class. Metadata and filenames are never forwarded to `SemanticFlow`.

`crop_events` maps `audio seconds + estimated offset -> actual frame PTS -> crop intersection`. Unresolved synchronization or events outside the probe's valid audio range retain content with no timing. A PTS/trajectory length mismatch fails closed. Roles are automatic audiovisual associations, not identity truth.

`SemanticFlow` calls the frozen source encoder first, then a trainable `SemanticBranch` outside its no-grad scope. The minimal branch averages learned UTF-8 byte embeddings of original English evidence; category and time are separate components. It is an interface baseline, not a pretrained semantic model. The public sampling interface accepts source tensors, typed weak events and explicit independent noise, with no listener/cache-path argument. NULL has a learned representation; semantic dropout is 0.1 in training. This is not an absence-of-reaction target.

Pilot commands, in order:

```bash
.venv/bin/python -m semantic_supervision.align.recompute_support
.venv/bin/python -m semantic_supervision.align.build_weak_cache
.venv/bin/python -m pytest -q semantic_supervision/tests
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m semantic_supervision.models.smoke_weak
```

The cache builder additionally expects the saved bounded lag probes; their exact no-network sandbox command and source-only input job are stored alongside results. The probe grid and gate policy are fixed before execution. The original 250-event artifacts are unchanged. The smoke run reads paired TRAIN targets only for the original FM loss, freezes T0, and updates the semantic branch for three steps. No checkpoint, FRC or FRD improvement claim follows from this check.

`auto_weak_experiment.json` records the later E0/E-text/E-event contrast. No formal budget is selected and no long training is launched. DEV must use the same frozen legal source-only pipeline. Existing transcript input must not be presented as an audio-only method.
