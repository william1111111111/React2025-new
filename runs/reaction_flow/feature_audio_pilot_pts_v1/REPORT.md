# Existing speaker feature / audio pilot: 33 TRAIN recordings

Executed CPU-only, without new APIs, visual inference, listener inputs, DEV tuning, or changes to the ongoing controlled experiment. Outputs are diagnostic candidates, not verified synchronization and not new eligible semantic labels.

## Results

| Visual signal | Audio signal | Median maximal correlation (32 computable recordings) | Candidates / 33 |
|---|---|---:|---:|
| Anonymous all-15 AU switching | Smoothed log RMS | 0.334 | 8 |
| Anonymous all-15 AU switching | Absolute envelope derivative | 0.222 | 3 |
| Global 52-D expression motion | Smoothed log RMS | 0.386 | 11 |
| Global 52-D expression motion | Absolute envelope derivative | 0.211 | 2 |

One recording has insufficient/constant evidence. These four diagnostics are separate, not a multiple-testing-corrected union. No model performance conclusions follow from these numbers.

All AU values are binary occurrences, not continuous intensities. Exact dataset AU order remains unresolved. No columns are described as mouth AUs. Anonymous per-channel diagnostics are saved, but no best-channel selection is used to claim a mouth mapping. The first 52 coefficients are used as global expression motion; last 6 pose coefficients are excluded. No learned projection was fitted.

## Critical timebase finding

All 33 facial-feature frame counts equal speaker video frame counts; measured video PTS follow approximately 30 fps, not the illustrative 25 fps. Initial 25-fps diagnostic is preserved in `../feature_audio_pilot_v1`, explicitly invalidated by `INVALID_TIMEBASE.md`. Its script snapshot and hashes are retained. That run cannot support offset conclusions.

The corrected run associates each feature row with the corresponding actual video PTS, conditional on the one-row-per-video-frame extractor assumption. Equal counts do NOT prove that assumption or validate synchronization. `video_grid_matches` in corrected SUMMARY only reflects that assigned PTS grid, not an independent synchronization test. No feature extractor timestamp manifest was located.

Lag convention is corr(M(t), A(t + delta)), so feature_time = audio_time - delta. Search ±1 s at 40 ms spacing; this grid resolution is not a precision claim. Derivatives, smoothing, articulation and audio energy can introduce physiological/signal-response lag, so a peak is not automatically a technical capture offset.

## Fixed diagnostic policy

Policy saved before each run: common noncircular support across all searched lags; at least 125 paired samples; positive peak correlation ≥0.2; nonboundary peak; advantage ≥0.03 over peaks ≥0.2 s away; both half-record peaks interior and within 0.12 s of the full peak; advantage ≥0.05 over reoptimized ±3/5/7 s noncircular controls. These are engineering assumptions, not calibrated probabilities. Control shifts have different valid support and are diagnostic, not a formal p-value. No thresholds were relaxed after seeing results. Audio is 10-ms RMS, transformed by log1p and smoothed at 80 ms; visual absolute first derivatives are averaged over channels and smoothed at 80 ms. AU second derivatives, VAD and phoneme activity were not evaluated.

## Agreement and limitations

18 recordings have prior TalkNet lag files; 11 are estimated, 7 unresolved, and 15 of the 33 have no lag file. Absence is not zero lag. Among the 11 estimated cases (including feature diagnostics that fail gates), absolute offset discrepancy medians are 0.16 s for AU/log-RMS and 0.08 s for expression/log-RMS; 6/11 and 9/11 are within 0.2 s respectively. This is descriptive agreement, not independent ground truth. TalkNet used a coarser ±0.4 s grid and role-supported event windows; current features use whole-record evidence, so support differs.

Only 4 accepted feature candidates per log-RMS method overlap estimated TalkNet cases. Their signed differences (feature minus TalkNet offset) are AU: -0.16, +0.72, -0.08, +0.04 s; expression: -0.08, -0.16, -0.08, -0.28 s. This small overlap does not establish robust agreement.

Five recordings pass both AU/log-RMS and expression/log-RMS gates, but only three agree within 0.2 s; the other discrepancies are 0.32 and 0.88 s. Consequently no candidate offset was injected into training or caches. Neither proxy establishes who is speaking.

## Mapping evidence

Local baseline README and baseline paper define 15 AU occurrences + 2 VA + 8 expression probabilities, and 52 expression + 3 rotation + 3 translation coefficients. Local FaceVerse coefficient concatenation is consistent with the latter. Baseline paper cites ME-GraphAU and GRATIS for AU extraction.

- https://github.com/reactmultimodalchallenge/baseline_react2025
- https://github.com/CVI-SZU/ME-GraphAU

ME-GraphAU README lists separate BP4D and DISFA label sets. Their union is NOT proof of the stored dataset's 15-channel order. Local baseline/current code and public upstream tree searches did not establish the exact dataset extractor mapping. Mouth-specific channel choice remains pending this evidence.

## Validation / next boundary

Three deterministic contract tests pass: known shifted signals with nonzero time origin recover the correct sign; constant/NaN signals remain unresolved; insufficient noncircular overlap cannot pass. Each recording saves input SHA256 hashes, curves, split-half peaks, control peaks, gate booleans and rejection reasons. SUMMARY records policy/job/script hashes. The first bootstrap failed on coefficient shape (T,1,58); this singleton axis was explicitly handled before the completed runs.

The next useful step is to obtain the actual AU extractor channel order or a documented FaceVerse mouth-landmark projection, then evaluate a preregistered mouth-specific signal and role-supported intervals. Current evidence supports an inexpensive diagnostic, not replacing TalkNet or declaring synchronization verified. Existing gold/auto_weak contracts, controlled training and official evaluation remain unchanged. Nothing was pushed.
