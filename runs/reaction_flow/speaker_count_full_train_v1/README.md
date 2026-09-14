# Full TRAIN voice-count screening

Scope: all 1,660 WAV files under `data/train/audio/speaker`, at most two sustained voices. Twenty-four CPU workers, one Torch thread each, zero GPU allocation; local public Silero VAD + ECAPA embeddings, no media upload or API calls. Existing training continues separately.

`counts.csv` and `SUMMARY.json` are generated at completion. `progress.json` is atomically updated while running. `records/*.json` contains every candidate voice window, VAD interval, short unassigned segment, graph sensitivity result and conservative distance diagnostic. `embeddings/*.npy` retains reusable speaker embeddings. No ASR or forced alignment was rerun, and full-corpus word attribution is outside this count-only step.

Buckets:
- `1_sustained_voice_candidate`: all tested graph settings favor one sustained voice.
- `2_sustained_voice_candidate`: stable two-way graph partition with minimum two windows and 2.4 seconds per group.
- `uncertain`: insufficient speech, unstable count/partition, or inadequate second-cluster support.

These are automatic candidates, not human-verified numbers of people. Even stable one-voice cases may contain a brief second speaker. Segments shorter than 0.8 seconds remain unassigned. No overlap detector is used. The method cannot establish source/listener identity or validate synchronization. No semantic training eligibility is granted.

The pilot's initial conservative distance gates failed to resolve all 33 examples. The subsequent exploratory spectral method is frozen for this expansion, including neighbor settings 0.2/0.3/0.4, k limited to 1–2 and stability requirements. Full-run classification replay matches all 33 saved pilot examples; this verifies code consistency, not real-world accuracy. Human review remains omitted, as requested.

The queue supports restart into the same output only with identical policy/code; existing records require matching audio/policy hashes. Each result is written atomically. Failures are explicit rather than assigned a speaker count. Prior pilot results are preserved. No git push is performed.

## Completed result

1660/1660 completed; zero errors; elapsed 175.3 seconds. Stable sustained-voice candidates: 1296 one-voice, 25 two-voice; 339 uncertain. Fresh full-run inference reproduces all 33 pilot classifications. These remain uncalibrated automatic estimates, not confirmed person counts.
