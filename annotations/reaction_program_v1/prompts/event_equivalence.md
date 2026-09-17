# Source event equivalence — prompt v1

Compare two annotated TRAIN source events from different recordings. Only source transcripts, source prosody/visual evidence and local source context are provided. You do not see session IDs, recording filenames, listener reactions, target trajectories or metrics. Candidate pairing is not evidence of equivalence.

Assess separately: communicative intent, dialogue stage, compatible local context, semantic content, ambiguity and speaker-role support. Same question wording alone does not prove equivalent context. Repeated text, differing negation, turn-taking, omissions and unknown source roles can make the relation UNKNOWN. Never infer equivalence because listeners react similarly.

Return compatible / incompatible / UNKNOWN, exact supporting quotes and supplied source evidence IDs for each side, same_intent / same_dialogue_stage / compatible_local_context (true/false/null), explicit uncertainty reasons. Optional numeric equivalence/confidence is subjective, NOT calibrated probability or a default loss weight. Do not invent timestamps or listener actions.
