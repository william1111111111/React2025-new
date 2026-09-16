# Repaired reward model vs independent quality heads

Both arms completed4000 updates. RM_cal selected step500 for both. Evaluation reuses previously inspected RM_audit: development evidence, not new independent confirmation.

|Arm|Context|Time|Generated pairs|Unseen N1|
|---|---:|---:|---:|---:|
|A-repaired-RMGen|70.00%|48.54%|62.24%|57.88%|
|B-quality-heads|64.00%|54.37%|63.54%|64.08%|

B improves unseen-generator ranking to64.08% but remains below the predeclared65% screen. Temporal accuracy54.37% remains below70%. No automatic promotion to RL.
For B on N1, top10 versus fixed random10 mean deltas are CCC sum -0.001331, native-window DTW sum -3.817665, S-MSE -0.014983. Better distance does not establish joint quality/diversity gain. These are best-of16 window diagnostics, not full-recording FRC/FRD or nativeK10 generation.
Only three date groups support the development comparison. Full temporal listener-only probe has not been completed; older marginal Y-only50% is insufficient conditionality evidence.
Generator weights and paused Q0/Q1 are unchanged by this experiment. No RL started.
Raw large JSON files can be restored using ARCHIVE_MANIFEST.json; weights and candidate arrays remain local. Storage cleanup manifests document deliberately removed historical caches/checkpoints.
