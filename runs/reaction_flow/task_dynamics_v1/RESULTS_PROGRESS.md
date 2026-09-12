# Task and dynamics adaptation: actual development evidence

A12000 has completed evaluation: FRC80 0.41471285389293405, FRC20 0.4159305496539825, exact FRD20 154.00543654725146 (2000/2000 pairs), S-MSE80 0.13943853974342346, FRVar80 0.06869899481534958. It improves distance relative to archived MAM but does not meet the joint FRC/FRD objective. No A retraining or repeated DTW was performed.

Cached-array analysis separates DC, P8-minus-DC and residual fast spread using identical full-time denominators. A12000 has approximately 34% fast spread versus 55% at A6000. Prediction-only P8 changes A12000 FRC from 0.414713 to 0.475945; this diagnostic does not replace raw main outputs. MAM P8 diagnostic is 0.851668; R2 is 1.198242. Fixed figures show all ten candidates at source0/20/40/60 with block boundaries. The double-precision algebraic spread differs slightly from legacy FP32 S-MSE numerics; raw official metrics remain unchanged.

Four fixed sources show no consistent decrease in Euler32-to64 output deltas relative to Euler16-to32. Higher NFE is not established as a remedy, nor is solver convergence claimed. Production and differentiable rollout remain Euler16. Fixed TRAIN FP32/FP64 task discrepancy is 1.1920928955078125e-7; max output discrepancy 8.069286207390558e-6.

Same-session cyclic donor substitution changes output but correct speaker inputs do not consistently improve paired CCC on these four probes. This identifies weak conditional correspondence as unresolved; it does not prove conditional independence. No conditioning projection is added in this experiment.

TRAIN16 calibration sets lambda_ref=1.1338064670562744 and lambda_dyn=0.17832674086093903. One of16 batches has no nonzero dynamic gradient and is excluded from the dynamic ratio median. Both task arms inherit A AdamW state and set lr2e-5, share records12001..14000 and all random streams, and retain the frozen encoder. T1 alone adds endpoint-difference reconstruction. The first200-step ramp is recorded in PROTOCOL.json. No clipping of gradient-based calibration coefficients or DEV tuning is used.

Actual new training and evaluations are recorded in arm directories and monitor_latest.json when launched; no T0/T1 result is claimed here. Old A, R2, S1 and MAM results remain intact. Training smoke and resume status must pass before formal launch. No push authorized for this task.
