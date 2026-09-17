# Paired event–observation association — prompt v1 (TRAIN only)

Input: already frozen source event evidence and separately obtained real paired listener observations. Preserve their original time axes. Report temporal association with observed delay/range, overlap, alternative explanations, and UNKNOWN if role, synchronization or event correspondence is unresolved. Temporal co-occurrence is not proof of psychological causation.

No moving event anchors to improve a listener match; no retiming raw paired trajectories. Each observed_support entry must reference a real observation_id and an evidence-backed association. Cross-recording support additionally requires a source-only compatible equivalence record. Transfer program support only; no donor waveform/trajectory endpoint or absolute timestamp transfer. LLM-inferred possibilities must remain separately marked, excluded from the first version's observed-support training.

Return event_id, observation_id, relation (temporal_support/UNKNOWN), delay interval and its time-domain conversion evidence, provenance and uncertainty. Never report a fabricated emotion, exact AU/3DMM, verified identity, or human review.
