# R2-cont versus R3-cover: fixed2000-step continuation

Both inherit R2 step6000 model, AdamW optimizer, RNG, normalization and loss scales. No optimizer reset or learning-rate change. Seed123; GPU1; T750/B4/K4; FP32 deterministic; lr1e-4 wd.01. Budget each2000 NEW steps, global6001..8000, checkpoints global6500/8000 (additional500/2000). The full seeded schedule is regenerated through8000 and first6000 exactly verified; only its tail is consumed. Alternatives remain full-source linear alignment then source crop, not the evaluation Processor and not causal frame labels.

A R2-cont: original valid+.25softmin-cover+.25paired+.5pairedES.
B R3-cover: replace only cover with detached minimum of24 permutations of4 slots, gather original differentiable costs, mean over candidate then input. Costs/weights/softmin temperature unchanged. Ties follow enumeration order; identical/duplicate target slots retain their declared weights and do not imply independent modes. Valid and paired-ES still supervise all sampled candidates.

Evaluation: native continuous AU; SAME Development80 source/ten-target/Processor/noise K10/750-block protocol, no reranking. Additional500/2000: FRC80/FRC20/S-MSE80/FRVar80, grouped differences/activity and CCC coverage from saved matrices. Final additional2000: same complete exact FRD20 (2000 candidate-target pairs per arm). No interim FRD search, no sample-bank expansion. Scores for a row always share its checkpoint/export policy. Parent exact FRD is reused, not recalculated.

MAM native archive is a reference with unequal pretraining/budget/AU rounding; not an architecture-causal baseline. Thresholds FRC80>.810962318916204 and FRD20<172.57642347297096 plus effective diversity toward MAM S-MSE=.15720370411872864 are development criteria, not forecasts or equivalence tests. FRC20 uses the same20 source population as FRD20. No independent confirmation claim.

Stop after both fixed budgets and final evaluations. If coverage replacement fails the quality/diversity tradeoff, retain all results and stop this variant; no automatic flow/codec/teacher or new loss search.
