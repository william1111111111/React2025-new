# Full local TEST evaluation

User requested TEST rather than VAL. This run pins P2 and both NFT step1000 checkpoints before scoring; training and scheduled DEV evaluations remain unchanged. The local official ReactionDataset contains 1142 directional cases (571 original speaker sources plus 571 reverse directions). Both directions are evaluated, and results are split by direction. For reverse cases the original listener is the legitimate source and the opposite original speaker is the target, exactly as the official loader assigns them. No target is passed to generation.

All 3426 local TEST audio-feature, 3DMM and facial-attribute arrays match VAL byte-for-byte. SPLIT_CONTENT_AUDIT.json preserves every hash. These results MUST NOT be described as independent hidden-test generalization. This is a complete evaluation of the local TEST snapshot.

No legal TEST semantic cache exists: every source uses NULL; no VAL event annotation is reused. P2 is re-evaluated under the same NULL condition, so comparison does not mix its historical DEV80 score with this setup.

Fixed K10, native standard recording noise, Euler16 FP64 -> FP32 official metrics. Targets follow official paired+9 sampling with full opposite-role same-session pool, fixed index-based seeds. Official Processor runs once per source with a fixed independent seed and shared output across all three models. Target processing does not depend on predictions.

Metrics: FRC, full exact FRD, S-MSE, FRVar, temporal S-MSE, TLCC, and the available MAE diagnostic. Official FRC sums each of K10 candidates' best target CCC; FRD sums each candidate's minimum target distance, then averages sources. Every FRD distance uses unsmoothed full-frame native25 DTW grouped as AU/15 + VA + expression/8. All 1142*100 = 114200 pairs per model are computed; no FRD20 extrapolation. Pair entries are atomically persisted in per-source matrices for restart. Only complete population results are published. Raw exports and per-source metric rows support re-audit.

Queue: prepare manifest/PTS -> shared target Processor -> three concurrent inference/metric workers -> three CPU FRD workers per model -> COMPLETE_RESULTS.md/json. GPU0 is reused within existing authorization; no new paid services or model downloads. Stage heartbeat/exit files and supervisor.log preserve errors. VAL extension was stopped with prior artifacts retained. No automatic push.
