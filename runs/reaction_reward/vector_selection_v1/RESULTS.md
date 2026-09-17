# Frozen RM vector selection

Reused RM_audit development diagnosis; 96 windows / 3 recording-date groups. Native-window best-of16 selection, not full-recording FRC/FRD or native K10 inference. No training or generator changes.

|Generator|Selection|CCC sum ↑|Native DTW sum ↓|S-MSE ↑|Fallback|
|---|---|---:|---:|---:|---:|
|P2|first10|0.611200|85.546258|0.065863|0.0%|
|P2|A-context-temporal|0.630448|84.779423|0.064193|0.0%|
|P2|B-context-temporal|0.659109|85.621224|0.063029|0.0%|
|P2|B-quality-CD|0.656333|86.039771|0.060714|0.0%|
|P2|vector-train|0.648280|85.770058|0.060946|100.0%|
|P2|vector-pool-first10|0.656508|86.016201|0.060710|100.0%|
|N0|first10|0.475424|94.765111|0.123710|0.0%|
|N0|A-context-temporal|0.473443|91.859102|0.114188|0.0%|
|N0|B-context-temporal|0.504271|93.714658|0.117738|0.0%|
|N0|B-quality-CD|0.482755|91.685080|0.111310|0.0%|
|N0|vector-train|0.482755|91.685080|0.111310|100.0%|
|N0|vector-pool-first10|0.482755|91.685080|0.111310|100.0%|
|N1|first10|0.479759|93.832858|0.118954|0.0%|
|N1|A-context-temporal|0.474235|90.283803|0.107440|0.0%|
|N1|B-context-temporal|0.506209|92.101229|0.112140|0.0%|
|N1|B-quality-CD|0.485243|90.280425|0.105436|0.0%|
|N1|vector-train|0.485243|90.280425|0.105436|100.0%|
|N1|vector-pool-first10|0.485231|90.283078|0.105432|100.0%|

Readout comparisons: `READOUT_COMPARISON.json`. Quality deltas versus same-pool first10 and fixed P2 first10, false-pass numerator/denominator, fallback slots: `SUMMARY.json`; group results: `PER_DATE.json`.

Predicted gates separately compare CCC and transformed distance: fixed RM_fit means, or same-pool predicted first10 means. Native-channel squared-distance greedy selection; quality-ranked fill when fewer than ten pass. A fallback is NOT quality-approved. False pass means predicted dual pass but failure of at least one corresponding independently computed true-quality threshold; it is not human preference error.

Same-pool gates do not restore P2 quality. Fixed P2 first10 comparisons are explicitly separate. There is no audit-based threshold fitting, no calibrated probability claim, and no independent confirmation from this reused split.

Inference reads source feature arrays/source PTS and candidate arrays only, plus frozen TRAIN normalization. Candidate time uses the source native grid, not listener PTS. Historical candidate pools and windows were previously constructed using targets; this is a fixed-pool diagnostic, not a proof of deployment sampling.

## Gate errors and fallback

|Generator|Gate|False pass / predicted pass|Fallback slots|Sets identical to B top10|
|---|---|---:|---:|---:|
|P2|vector-train|54/103 (52.4%)|89.3%|80/96|
|P2|vector-pool-first10|264/361 (73.1%)|62.4%|95/96|
|N0|vector-train|9/14 (64.3%)|98.5%|96/96|
|N0|vector-pool-first10|256/357 (71.7%)|62.8%|96/96|
|N1|vector-train|4/11 (36.4%)|98.9%|96/96|
|N1|vector-pool-first10|251/369 (68.0%)|61.6%|95/96|

## Deltas against fixed P2 first10

|Generator|Selection|Delta CCC sum ↑|Delta DTW sum ↓|Delta S-MSE|
|---|---|---:|---:|---:|
|P2|first10|+0.000000|+0.000000|+0.000000|
|P2|A-context-temporal|+0.019248|-0.766835|-0.001670|
|P2|B-context-temporal|+0.047909|+0.074967|-0.002834|
|P2|B-quality-CD|+0.045133|+0.493513|-0.005149|
|P2|vector-train|+0.037080|+0.223800|-0.004918|
|P2|vector-pool-first10|+0.045308|+0.469944|-0.005154|
|N0|first10|-0.135776|+9.218853|+0.057847|
|N0|A-context-temporal|-0.137757|+6.312844|+0.048325|
|N0|B-context-temporal|-0.106929|+8.168400|+0.051874|
|N0|B-quality-CD|-0.128445|+6.138823|+0.045447|
|N0|vector-train|-0.128445|+6.138823|+0.045447|
|N0|vector-pool-first10|-0.128445|+6.138823|+0.045447|
|N1|first10|-0.131441|+8.286600|+0.053091|
|N1|A-context-temporal|-0.136965|+4.737546|+0.041577|
|N1|B-context-temporal|-0.104991|+6.554972|+0.046277|
|N1|B-quality-CD|-0.125957|+4.734167|+0.039573|
|N1|vector-train|-0.125957|+4.734167|+0.039573|
|N1|vector-pool-first10|-0.125969|+4.736820|+0.039569|

## Interpretation

B quality readout improves unseen-N1 pair accuracy to 64.08%, versus B context+temporal 61.76% and A 57.88%. All 288 pools fall back under both gates: fewer than ten predicted jointly eligible candidates. Thus diversity selection mostly reproduces quality top10 rather than demonstrating a successful quality-diversity tradeoff. B context+temporal has higher selected N1 CCC and diversity than B quality readout, while B quality readout has lower distance. There is no uniformly best readout. N0/N1 remain lower-CCC and higher-distance than fixed P2 first10 despite greater diversity. Keep checkpoints frozen; no evidence here supports automatic RL promotion or threshold relaxation.
