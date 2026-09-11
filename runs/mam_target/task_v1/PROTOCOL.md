# MAM-target first batch (development, seed123 only)

Goal: same checkpoint/policy FRC > archived MAM .810962318916204 and exact FRD20 <172.57642347297096, with valid multi-channel diversity. Engineering milestones .85 /160 are goals, not observed results. No new seeds, flow training or automatic hyperparameter search.

R1/R2 start from scratch, same small-nonzero projection std .01, pre-norm, original nonzero FiLM, standard_normal. Residual logits have no inner tanh. Both share full Phase25 seed123 schedule/noise/crops and 6000 actual steps, T750 B4 K4 AdamW 1e-4 wd .01 FP32. Fixed evaluation checkpoints2000/6000. R1=paired ES + .1 C2 descriptor ES. R2=valid +.25 cover +.25 paired +.5 paired ES, with no descriptor term.

R1 vs old C2 changes BOTH the residual function and projection initialization. It is a versioned new-head branch comparison, not a clean attribution to removing tanh alone. R1 vs R2 differs in supervision, with exact same initial model hash and data/noise streams.

TRAIN R2 targets: paired+3 same-session alternatives, unique excluding paired if >=3, otherwise replacement. Selection deterministic from seed123/step/occurrence/source. Paired uses actual overlap. Weak alternatives are linearly resized over the FULL source duration before taking the SAME source crop. This is explicitly TRAIN-only weak alignment, not the official learned Processor and not causal frame supervision. Reference input cannot enter the generator. Record paths/IDs/crops/lengths per row.

CCC uses original 25-channel population formula, including constant-channel zero convention. Soft-DTW divergence uses squared Euclidean group costs with weights1/15,1,1/8; full unrestricted soft recurrence on uniformly linearly resampled32-point valid grids, gamma .1, divided by32. This is a training surrogate, NEVER exact FRD. Anti-diagonals are batched across all KxR pairs. Softmin temperature .1 normalized by candidate/target count. No OT/Hungarian/DPP or free diversity reward.

Task coefficient a=1; b fixed by median CCC/SDTW prediction-gradient norm ratio on first3 TRAIN batches. beta/gamma/eta fixed above before development scores. This is an initial gradient scale calibration, not proof of balanced objectives; curves/logs remain necessary. Paired best-match CCC+L1+.1 velocity L1; all samples also receive paired ES and task-valid supervision.

Evaluation uses existing Development80, same frozen10 targets, original Processor cache, K10 frozen bank, global epsilon across750 blocks, original FRC/FRVar/S-MSE/TLCC, exact same FRD20 subset. Native continuous AU primary; any .5 threshold diagnostic will be separately named. No GT reranking. No hidden test claims. Multi-seed remains paused.

Run resources: GPU0 R1/R2 parallel after real shape/resume gates; evaluation can overlap with training if memory permits. Exact FRD uses CPU, 2000candidate-target pairs/model/point, complete-only means. No other processes stopped. Historical outputs read-only; no push.

Sources: official metric files at /home/zhengshiyi/react2025/framework/metrics; Soft-DTW divergence https://proceedings.mlr.press/v130/blondel21a.html . All implemented components are established tools, not novelty claims.
