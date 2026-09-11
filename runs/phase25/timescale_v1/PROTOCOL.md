# Phase25 timescale alignment v1

Frozen before new training/evaluation. Baseline 7e6adf562d628aff78a9e22d74a111def7fa3353.
HiRP22 unchanged, standard_normal, pre-norm/default projections, FP32, AdamW 1e-4/.01.
C0 lambda=0 effective; C1/C2 lambda=.1. T750, B4 independent source occurrences in two banks, K4.
Seeds (init,sampler,noise): (123,123,123), (42,42,42), (2026,2026,2026).
6000 steps each; first execution segment seed123 all arms through2000. Checkpoints128/500/1000/2000/4000/6000; intervals500, diagnostics100. No new lambda/noise evaluation grid.

Source sessions proportional to number of sources. Source occurrences iid with replacement within session. Source-only uniform crop start in [0,max(0,source_total-T)], deterministic SHA256 of seed/occurrence/source path (existing PairedReactionDataset law). Paired listener uses same start; target length is not used in crop selection. Empty pair raises without resampling; failure is counted. References are unique, up to8 uniform without replacement, identical C1/C2, independent crop starts seeded by sampler seed/step/reference ID with separate prefix. Only raw arrays cached (LRU48); no occurrence-specific descriptor/item cache.

Reference marginal: p(session)=source_count/N, uniform unique references, uniform legal crop. New train-only scaler uses four independent fixed draws per reference, seed25000, weighted by this reference marginal. Source-paired and all-reference finite populations are distinct and not claimed identical. Source and reference clip length750; all arms use one shared new scaler. Raw-unit conditional ES stays25 unit channels. Historical T128 descriptor scores remain a distinct protocol.

Existing Phase24 16 models: official sum max CCC unchanged; auxiliary generated-side mean, target-side mean, maximum one-to-one mean preserve10 slots including repetitions. No GPU generation for this analysis.

New task evaluation uses frozen Phase24 Development80 source/target/noise manifests and processed-target files, K10 and750 blocks with one global epsilon per sample index. Old cache hashes verified; new namespace records current complete source snapshot. Confirmation remains pending authoritative identity mapping.

Exact FRD plan:20 sources, lexicographically first clip_id per session, fixed without score access. Seed123 C0/C1/C2 new2000-step models plus native archived MAM;100 candidate-target pairs each, three exact unrestricted tslearn DTWs per pair, full effective frames. Total8000 pairs/24000 DTWs. Atomic per-pair resume, initial one-pair timing; CPU budget up to2 hours this segment. Partial results are not a complete FRD mean. No banding/truncation/surrogate.
