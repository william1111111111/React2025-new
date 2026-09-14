# BERT semantic controls: execution record

Independent local branch `agent/bert-semantic-controls`; historical E-series outputs preserved. Original T0 step14000 parent SHA `e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6`. Four new arms: P0-null, P1-byte768, P2-frozen-BERT, P3-frozen-BERT with explicit frame-event bias. No event type, planner, shared noise, new teacher calls, or automatic push.

BERT is `google-bert/bert-base-uncased` at revision `86b5e0934494bd15c9632b12f734a8a67f723594`. Trusted safetensors loaded locally with remote code disabled. Encoder frozen and eval in a separate preprocessing process; 345 unique legal source texts cached FP32. Last hidden state content-token means exclude CLS/SEP/PAD; no pooler output. 128-token nonoverlapping windows and token-count weighted pooling handle overflow; no real input exceeded126 content tokens. Blank input stays missing. Cache replay max absolute difference across batching was1.67e-6; order probe difference is not semantic accuracy.

48 TRAIN sources and frozen DEV80 are reused with unmodified weak gates. DEV source-level strata:48 non-NULL,32 NULL;38 have timed events entering at least one crop. Some crops in non-NULL sources remain NULL. TRAIN/DEV proposal-distribution difference is preserved and reported. The 1000-record schedule reproduces the old first500 records, with a separately precomputed shared dropout mask. All four arms start at T0, not old E checkpoints. Each adds1000 actual optimizer updates, B4/T750/K4; velocity lr2e-5 with inherited AdamW, new semantic lr1e-4 warmup100, wd.01, clip1.0. No BERT weights are loaded during training.

Acceptance completed before launch:
- 5 implementation contract tests and8 supplied reference tests pass.
- All four step0 full750-frame predictions equal original T0 exactly in the measured FP64 check; K-prefix max error2.94e-15 and repeat error0.
- Real3-update P3 continuous vs1+resume-to3 gives identical model, optimizer and RNG state.
- First output-projection gradient0.02270084; upstream content-projection gradient begins after output leaves zero (third-step norm4.04e-6).
- After3 diagnostic updates, legal content substitution changes predictions (max0.00221); this is sensitivity, not a task-quality claim.
- Actual NULL fallback remains the trained arm's own base condition.
- Existing E0 exports were used ONLY to check aggregation math, never as the new P0 control.

A tokenizer special-mask API incompatibility and tuple/list log-replay mismatch were caught in bootstrap runs and fixed. Original bootstrap logs are retained in diagnostics. No failed bootstrap is presented as formal training.

`queue.py` runs serially on GPU7: train500 -> DEV80 curve -> resume1000 -> full DEV80 for each arm, then complete exactFRD20 (2000 pairs each), fixed P2/P3 text and time interventions, and a final joint report. Same-checkpoint main and intervention outputs have separate labels. Fixed interventions do not select inputs by targets. Unmeasurable sources are excluded from intervention summaries rather than counted as zero effects. Main reporting retains all80 sources, with NULL/non-NULL/timed strata, per-source paired differences, session-equal summaries and session bootstrap, target-best CCC, and DC/slow/fast spread.

Check `queue_status.json`, `training/<arm>/monitor_latest.json`, and `RESULTS.json`. A scheduled experiment is not a completed result. Checkpoints and vectors remain local ignored artifacts. Fixed code and artifact identities are checked on process entry; no automatic budget extension. The final report states current MAM reference and unequal-cost limitations.
