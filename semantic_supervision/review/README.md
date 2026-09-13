# Local review handoff

Run `python -m semantic_supervision.review.export` after extraction. It creates 200 unique source-only candidate windows, 50 double-review tasks (250 independent answer slots), and20 source-only cross-recording review tasks. No tasks are sent to people or external services.

Speaker, listener and paired tasks are separate files. Source reviewers receive only speaker material. Listener reviewers receive only listener video and numeric change proposals. Paired review happens after both independent event annotations and sync checks. Cross-recording matching sees only the two speaker event/transcript records, never listener attributes or model scores.

Task files use opaque recording tokens; `private_recording_map.json` is local operator-only and must not be sent to teachers. Reviewer payloads do not include session ID or descriptive original filenames. Media remain in their original local directories and were not uploaded or copied into this handoff.

Candidate windows are selected across TRAIN sessions and low/typical/high speaker-energy strata. They are NOT verified semantic events, silence/VAD labels, emotional categories, or proof of no reaction. Full transcript text is not assigned fabricated word times. Content, occlusion and ASR-difficulty stratification remain pending real review. Windows can overlap; they are not independent subjects.

For each window reviewers should verify role, audio/video offset, transcript content, event boundary intervals, visible observations, no-visible-reaction, occlusion, and ambiguous/unknown links. Fifty B-review slots must be completed independently before adjudication. Blank answers are not errors or agreements and must not be counted as completed annotations. Twenty matching tasks cover17 unique recording pairs;3 tasks revisit a pair with a different A-window focus. No initial semantic correspondence is asserted.

The material pack's schema checks format and declared provenance, not truth. `models/source_cache.py` additionally requires accepted provenance, registered evidence, matching media hashes and verified role/time before source labels can be consumed. All current records deliberately fall back to NULL. Do not manually set accepted flags merely to make a training run start.
