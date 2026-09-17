# Semantic Reaction Program — minimal implementation v0

Preparation lives in `preparation/`; data and prompts are under `annotations/reaction_program_v1`. No old generator, RM, GMM, Flow/Diffusion or planner weights are imported.

- `motion_codec.py`, `rvq.py`: 25D stride4, latent128, two residual codebooks of512; sigmoid AU / tanh VA / softmax expression. BCE AU, VA MSE, expression KL, active-channel CCC, block-balanced velocity and VQ/commitment losses. 58D auxiliary is not enabled. Valid-frame masks retain short tails; padding cannot contribute a target loss.
- `reaction_program.py`: explicit multi-action IDs, overlapping onset/peak/offset and observed intensity. No target trajectory, recording/session ID, or filename enters the executor. Pending/UNKNOWN records are rejected as real oracle training labels. The action vocabulary must be supplied separately; test placeholder names are NOT a learned ontology.
- `reaction_clock.py`: encodes oracle timing and local action envelopes. **Predicted speaker clock is not implemented**; it belongs after oracle feasibility.
- `motion_generator.py`: two-layer width128 masked motion-token Transformer. Program-global and time-local conditioning, separate RVQ-level readouts, masked-token CE. Eval sampling accepts explicit uniform streams and preserves K-prefix. No style noise is added in v0; no learned planner or forced rare-action allocator exists.

Tests: `.venv/bin/python -m unittest reaction_program.test_minimal reaction_program.preparation.test_contracts -v` (13 tests). Smoke: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m reaction_program.smoke`; exclusive output directory, no overwriting prior smoke results.

Executed smoke: two real TRAIN crops, 12 codec optimizer updates on CPU; one executor backward with an explicitly UNLABELLED placeholder program, zero executor optimizer updates. This checks tensor/mask/gradient integration only. It does not test learned semantic controllability. Total codec objective rose2.5668→3.7258; per-level active code count fell2→1 on this tiny batch. No quality gate passed, and no full validation metrics or oracle EXP-C are claimed. Wider TRAIN representation/initialization and masked reconstruction must be assessed before long codec training; do not use validation to tune an ontology.

Full-recording EXP-A/B/C requires restored exact Processor targets shared across all arms. Codec evaluation must compare decoded GT to raw GT on identical frozen references, with explicit oracle status. Program executor requires actual observed programs;600 pending media tasks are not600 training labels. Real labels, import/merge logic and fitted ontology are outstanding. Do not automatically train a planner or launch RL.
