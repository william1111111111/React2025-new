# Observable action ontology v0 — open, not fitted

Status: no new listener video annotations have been completed. There is **no claimed learned action vocabulary or frequency table** yet. This is the versioned discovery contract; preset example action names are not training classes.

Each action stores: raw observable phrase; region and movement description; independent onset/peak/offset or bounding ranges on native PTS; co-occurring action IDs; observed intensity with scale definition; occlusion and evidence frames; raw teacher response/model/revision/prompt/input hashes; unknown reasons. `UNKNOWN` is metadata, not an action token; maintain is an observation requiring an inspected interval.

Discovery uses only the first TRAIN-fit batch. Retain all raw phrases, embed/cluster only after visual descriptions exist, then record merge provenance and contradictory examples. A possible phrase cluster must be supported by real observations in multiple recordings; no session ID, model name, filenames or emotion-ground-truth label is supplied to the embedding input. Freeze merges/version before any validation evaluation. No test-driven merges. Annotation count and independent recording count must both be reported.

Compositionality: programs are sets/partially ordered graphs of actions, with temporal edges and optional observed style; no mutually exclusive 20-way emotion classifier. Keep lexical synonyms separate until an evidence-backed merge exists. Numeric detector windows only propose where to look and are not proof of an action name.

Strict / relaxed: strict requires source role and time support, direct visible observation with adequate coverage, valid native frame bounds and traceable paired association. Relaxed retains uncertain timing/role/occlusion for analysis with explicit reason fields and no default precise supervision. Confidence alone never moves a record between regimes. Neither regime automatically becomes human gold. Human spot checks requested by the latest taskbook will be stratified by self-rated confidence, rare phrase, cross-recording match and boundary ambiguity; no completed review is currently asserted.
