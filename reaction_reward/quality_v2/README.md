# Fresh quality-prediction A/B, stage A only

Authorized after review_v2. New output directory: runs/reaction_reward/quality_v2.
Two fresh seed123 models, no pretrained reward backbone and no continuation of
old4000-step rewards. A uses repaired RM-Gen ranking; B additionally predicts
independent standardized native-window bestCCC and minDTW targets. The extra
heads are instantiated in both models for identical initialization but frozen
and unused in A. Both see the same8 quality-input candidates per update;
A forward is diagnostic/no-grad, B has Huber quality supervision.

Real fit evidence expanded to1986 nonoverlapping windows from1100 eligible
recordings. Same original fold and fit normalization. Frozen nearest-PTS rules
and effective source/date-balanced sampler; unknown ranking labels stay masked.
Fit valid context1248, temporal1990, weak2637, generated2048. Generated quality
regression uses8192 candidates, including preference-unknown candidates.

Previously deleted arrays are reconstructed with exactly the old source/noise/
model config and must match the old SHA256 before becoming available. No new
DTW, no new candidate IDs. Maximum14848 recreated windows, counted as cost.
Queue waits for both restoration shards, verifies protocol hashes, launches
A/B in parallel and evaluates frozen RM_cal-selected checkpoints. Historical
RM_audit is reused development data, not fresh independent confirmation.

Max4000 updates, checkpoints250/500/1000/2000/4000. No RL/generator update,
no Q0/Q1 edits and no automatic push. Full temporal listener-only training is
not added as a third main arm; current conclusions must retain that limitation.
