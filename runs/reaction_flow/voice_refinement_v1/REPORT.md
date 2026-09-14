# Low-volume recovery and short-voice context follow-up

Executed on TRAIN speaker media only: 12 CPU workers for 95 low-volume cases and 8 CPU workers for all 79 small-cluster cases. Zero processing errors; no GPU, paid API, listener media, new visual model, or training/cache changes. Four deterministic gain/interval contract tests passed.

## Low-volume result: gain did not resolve speaker count

All 95 selected cases had insufficient voice evidence and RMS below -60 dBFS. A single gain was chosen from target RMS 0.05, a maximum +60 dB, and a peak ceiling of 0.95. The same frozen VAD/embedding/clustering settings were then applied. A second VAD pass at half that gain records gain sensitivity; thresholds were not adjusted to increase acceptance.

- Median gain: +36.88 dB. Maximum processed peak: 0.95000005 (float rounding).
- 9/95 produced any VAD speech proposal; maximum total proposed speech was only 0.612 s.
- 0/95 produced a >=0.8 s usable voice window. All 95 remain uncertain.
- 93 of these 95 come from session6, one from session0 and one from session7. Session metadata was examined after inference solely to locate a possible systematic issue; it was not a model input.

This shows that the tested bounded gain alone does not recover usable speech. It does not prove the recordings contain zero speakers, that VAD cannot fail, or that a high-gain signal is all noise. Peaks can limit gain, and near-floor PCM information cannot be reconstructed by multiplication. Median exact-zero sample fraction was approximately 56%. No denoising or source separation was tested.

Several sampled session6 source texts repeat “Thank you” or “Okay” despite the lack of sustained detected speech. This is a reason to audit the audio/transcription source, not proof that every such transcript is fabricated. These texts must not independently establish a speaking identity. `audio_evidence_flags.json` records missing acoustic support, with count unknown and `no_event=false`; it is a diagnostic list and is not wired into training automatically.

## Short second-voice result: context remains weak evidence

All 79 small-cluster cases were examined, covering 85 minority-cluster windows. Each has up to 2 s of preceding/following audio context. A flank receives a new embedding only if it is >=0.8 s and at least half of its duration was speech under the original VAD. No disjoint snippets were concatenated to invent a continuous turn.

- 43/79 recordings have at least one usable flank; 36 do not.
- 47/85 core windows have a usable flank, totaling 54 flank embeddings.
- 48/54 flanks are closer to the majority-cluster centroid than the minority centroid. This is descriptive context consistency, not evidence sufficient to prove that the core belongs to a second person. Short core embeddings can be affected by noise, phonetic content or a turn boundary; minority centroids also include their own core embeddings.
- Existing TRAIN ASD scores were available for only 3/79 recordings. They are saved as uncorrected legacy-time-grid diagnostics and do not establish identity.

Actual video PTS were extracted, with matching AU/3DMM row counts. Each core and flank includes anonymous all-15 AU switching and global 52-D expression activity. AU order is still unresolved, so these are not mouth-specific signals. The comparisons assume zero audio/video offset and equal-index correspondence only for descriptive statistics; no synchronization claim follows. Core ASD evidence varied across the three available records. No automatic source/listener assignment or speaker-count promotion was made.

## Artifacts

- `low_gain_comparison.csv`: original versus amplified speech duration and usable windows, gain and RMS for all 95.
- `low_gain/*.json`: full new VAD/clustering results, gain sensitivity, source and policy hashes.
- `context/*.json`: all 79 records' core/flank intervals, embedding similarities, AU/3DMM activity, and existing ASD where available.
- `SUMMARY.json`, `POLICY.json`, `MANIFEST.json`, `tests.txt`: counts, frozen assumptions, inputs and implementation checks.

The practical next priority is a source-media audit of session6 and, for short interjections, a change/overlap-aware diarization model or direct audio examination. This round does not justify forcing uncertain records into one- or two-person labels. Original full-TRAIN results and ongoing E0/E-text/E-event experiments are preserved. No git push was performed.
