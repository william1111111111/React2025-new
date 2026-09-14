# 33 TRAIN recordings: at-most-two-voice diagnostic

Executed locally on CPU. Downloaded public ECAPA-TDNN and Silero VAD weights; no dataset media upload, paid API, GPU use, listener features or cache changes. Existing transcript CTC alignment supplies words; no new ASR decoding was run. Voice clusters are local acoustic groupings, not person identities.

## Deliverable and interpretation

Of 33 recordings, **25 consistently favor one sustained voice under the tested graph settings; 8 remain uncertain**. This does not certify that those 25 contain exactly one person: brief interjections, overlapping voices and a weak second speaker can be missed.

Primary spectral settings return 28 one-voice candidates, 2 two-voice candidates and 3 insufficient-evidence cases. Of the 28 one-voice candidates, 3 change under graph sensitivity checks. Both two-voice candidates lack minimum sustained second-cluster support. No recording is confirmed two-person.

Two-voice candidates, both uncertain:

- `rec_3b7feda5f993fb04`, session17/Camera-2024-06-21-103121-103102: small second cluster and graph sensitivity.
- `rec_a0c982590e7adf5d`, session17/Camera-2024-06-26-093649-093642: small second cluster.

`counts.csv` lists every recording, candidate count, graph stability and uncertainty reasons. Each per-record JSON contains candidate voice time windows, existing aligned words assigned by ≥80% interval overlap, short unassigned segments and all graph results. Unassigned words remain unknown. No voices are named source/listener, male/female, or matched across recordings.

## Method and failed first diagnostic

Silero VAD identifies speech on the WAV sample axis. Nonoverlapping windows of 0.8–2 s feed SpeechBrain ECAPA-TDNN (VoxCeleb), using the published 80-bin filterbank, sentence mean normalization and L2-normalized raw encoder output. Segments shorter than 0.8 s remain unassigned. Windows are not guaranteed speaker-pure, and there is no overlap-speech detector.

The initial conservative cosine-distance/silhouette heuristic returned **33/33 unresolved**. Those results remain intact under `../speaker_count_pilot_v1`; they were not overwritten or presented as successful counts. We then performed a separately recorded exploratory spectral analysis on the SAME embeddings. This is a methodological follow-up, not independent confirmation or calibrated validation.

Spectral analysis forms a nonnegative cosine graph, retains row neighbor fractions 0.2/0.3/0.4, symmetrizes, and uses the unnormalized Laplacian's eigengap to choose k=1 or k=2. Including the first eigenvalue is essential to allow one speaker; the installed SpeechBrain diarization default starts at two speakers and was therefore NOT used as-is. Primary fraction is 0.3. Stable counts require agreement across all settings, two-cluster partition ARI ≥0.8, and ≥2 windows / ≥2.4 s for each of two clusters. Graph agreement is NOT a probability of correctness. Short unassigned segments remain an uncertainty even when graph stability is true.

No clustering parameter used filenames, session IDs, listener inputs, face motion, or DEV performance. Filenames in reports are traceability metadata only. User's at-most-two assumption caps the candidate count; the algorithm cannot establish absence of additional voices.

## Existing facial features

`feature_aux.json` describes anonymous AU switching and global 52-D expression activity within each candidate voice's intervals. Actual video PTS are associated with corresponding feature rows, whose counts match. Zero offset is used only for descriptive summaries; no synchronization or source-identity claim follows. AU order remains unresolved, so no mouth-AU selection is fabricated. No new mouth network was run.

The first diagnostic also saves existing ASD statistics; those use its legacy score time grid and are explicitly uncorrected auxiliary evidence. They do not resolve the uncertain voice identities. The present work completes initial voice-count screening, not validated full diarization or a new WHO/WHEN training cache.

## Provenance and checks

ECAPA revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` from `speechbrain/spkrec-ecapa-voxceleb`. Silero revision and downloaded metadata are stored in the external model cache. First-stage POLICY saves model/code/audio-job hashes, and each record saves audio/CTC/ASD hashes. Second-stage records hash their first-stage input. Dependencies were installed into a separate cache directory, leaving the training environment's packages unchanged.

Seven deterministic synthetic contract checks pass: single/two synthetic voices, insufficient speech, single outlier, nonoverlapping segmentation, one-speaker-permitted eigengap, and separated graph components. These check implementation, not accuracy on real recordings. Human review was omitted as requested; no human approval or measured diarization accuracy is claimed.

Next useful evidence would be a stronger pretrained change/overlap-aware diarization pipeline or better-supported longer speech segments, followed by independent checks on the ambiguous short second-voice candidates. Do not use this exploratory count alone to relabel all transcript words as source speech. Existing controlled experiments and official metrics remain unchanged; no push was performed.
