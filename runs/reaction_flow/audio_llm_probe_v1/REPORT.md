# Yunwu audio-input probe

Six small paid inference requests were executed: Flash-Lite on three real TRAIN speaker recordings and 8 seconds of synthetic silence; Flash on one of those real recordings and the same silence. No listener media or local speaker-count predictions were supplied to either model. No full-corpus API job was started.

The OpenAI-compatible `input_audio` WAV payload is accepted (HTTP 200). The three real opening transcriptions agree with existing source text, providing evidence that audio content is processed. This does not verify the platform's underlying model identity or its speaker-count accuracy.

Flash-Lite reported one speaker for all three recordings, including one local two-voice candidate and one uncertain case. It also hallucinated a voice and the sentence “I'm not sure if I can do this” on an all-zero synthetic WAV. Therefore it failed the silence negative control and is not accepted as an automatic adjudicator.

Flash reported one voice for the disputed real recording. On the same synthetic silence it abstained (null count, empty transcription), but described the audio as missing/inaccessible rather than correctly recognizing silence. This is a non-hallucinating response on that test, not complete silence-understanding validation. Its usage returned zero modality breakdowns despite nonzero prompt tokens, so modality accounting is provider-dependent.

Flash-Lite real samples (duration / input / output tokens): 27.05 s / 869 / 125; 17.95 s / 619 / 105; 18.85 s / 644 / 114. Flash real sample: 17.95 s / 872 / 122. All six requests together reported 4,655 tokens. Currency cost was not available in responses and is not inferred from public vendor pricing.

The WAVs were downmixed, resampled to 16 kHz and gain-adjusted with capped gain and peak protection. Only those derived inputs were uploaded; original files remain unchanged. Hashes, exact prompts, responses, durations, gain and usage are saved. No transcript, voice labels or training eligibility was rewritten. Automatic provider/model judgments remain diagnostic.
