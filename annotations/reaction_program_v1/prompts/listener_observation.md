# Listener observation annotator — prompt v1

You receive a TRAIN listener video, its native frame PTS, and optionally numeric trajectory evidence. Describe what visibly changes in open vocabulary; do not receive a presumed desired emotion or program. First annotate observations independently of speaker events. Later an association task may link them, retaining uncertainty.

Use short anatomical descriptions such as motion direction, body region, temporal order, co-occurrence and visible intensity. These are examples of description structure, not a fixed vocabulary. Multiple simultaneous actions are allowed. Maintain/no-new-change is valid ONLY when a visible interval was actually inspected. No video access, occlusion, failed detection or missing labels must produce UNKNOWN, not maintain.

The 25D arrays comprise 15 anonymous AU occurrences, 2 VA channels and 8 expression probabilities; exact AU/class ordering is unresolved. Do not name anonymous columns as specific muscles. The first 52 of 58D are expression coefficients, not named actions. Never generate, interpolate or replace exact AU/3DMM values; the numeric target always comes from the original array. A sustained expression level is not evidence of speech activity.

Use supplied frame IDs and real PTS for onset/peak/offset. If frame access is sparse, return bounding intervals and uncertainty, not a falsely precise time. Keep every action's own interval; overlapping actions need not share an onset. Do not use the audio to infer the person's hidden intention or personality.

Return observation_id; status (observed/UNKNOWN); actions (open phrases + onset/peak/offset evidence or uncertainty intervals + ordinal observed intensity and justification); overall visible motion description; coverage/occlusion; accessed_modalities; evidence frame IDs; subjective uncalibrated confidence; uncertainty_reasons. Return null actions if evidence was inaccessible, not an empty list. Empty actions means an inspected no-new-change interval and requires coverage evidence. No event correspondence or program equivalence is presumed. Never assert human_reviewed=true.
