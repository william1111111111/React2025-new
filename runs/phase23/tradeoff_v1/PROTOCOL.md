# Phase 2.3 locked finite protocol

Question: does cross-input marginal matching improve group fit under a controlled conditional-quality constraint?

Keep HiRP22 pre-norm/default, A0, the original all-sample conditional ES and C1/C2 group estimators. Three seeds 123/42/2026, 2000 actual steps, T128/K4/B4, AdamW 1e-4/wd .01. Reuse C0 and lambda .1 after fingerprint/config/schedule checks. At most 12 new scratch trainings: C1/C2 × lambda .03/.3 × three seeds. Finish seed 123 before other seeds. No extensions or checkpoint mixtures.

Final frontier: both fixed banks averaged at K32, K64 stability. Curves: bank0 only, explicitly labelled. All discrete points remain visible. Relative conditional ES ≤ C0×1.01 is a post-Phase22 engineering screen, not preregistered historical evidence or statistical equivalence. Report individual seed differences, uncertainty and failure to match; do not interpolate points.

Development-80 remains the only source evaluation population. Confirmation candidates stay pending and unused. Six exhaustive 2-of-4 source mixtures per session assess finite source-subset sensitivity against the same full reference pool. These are not independent datasets/seeds. Raw covariance and lag 1/4/16 autocorrelations are fixed evaluation-only summaries with no validation scale fitting.

Task integration uses the existing nine checkpoints first: K10, full source records, zero-padded T750 chunks, fixed global epsilon per sample index across all time chunks. Entire valid source sequences are exported; no silent time truncation. Time-chunk and full-context predictions need not agree. The old official chunk convention creates an extra zero-valid tail at exact multiples: retain its padding layout but do not call the generator on zero-valid chunks; trim that tail on reassembly. Attribute metric implementations and target alignment processor stay unchanged. Targets are the single paired development listener; therefore this is a development full-sequence protocol, not the official ten-appropriate-target benchmark. FRC metric matching is used as implemented, never to rerank predictions. FRD/rendering/video-realism metrics and external MAM comparison are outside this bounded initial integration.

Cache reuse requires the complete content/implementation identity and result hash. Missing identity, different inputs/configuration or tampering fails. Full reruns use a new evaluation-name directory. Repeating a fully completed identical invocation is a no-op.
