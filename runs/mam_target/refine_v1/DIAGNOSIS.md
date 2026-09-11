# Read-only baseline diagnosis and bounded decision

All five baseline model/point results were verified from existing exports and 10,000 cached exact FRD pairs; no new generation or DTW. RESULTS_LATEST.json stores full checkpoint/evaluator/target identities. The old comparison snapshots were not edited.

R2 step6000: FRC80=1.1814686849299616, FRC20=1.134674339994814, exactFRD20=133.40907425912587, S-MSE80=.04367993772029877, FRVar80=.039064668118953705. Parent SHA256 d9b0e07aad91f3bbb254bfc2ad5780f9fa2b9a138245dbafd0c79fa935bf8936. It remains the user's fixed parent.

R2's within-input offdiagonal candidate MSE by AU/VA/expression is .0600653/.00342924/.0230262, versus MAM .181608/.118666/.121086. Centered differences are .0392671/.00234834/.0129521 versus .0968363/.00673608/.0236816. Both mean-level and temporally varying candidate differences are reduced; VA mean-level differences are particularly limited. These statistics cannot separate semantic event timing from amplitude by themselves.

No candidate pair in the audited per-group records met the explicitly diagnostic MSE<1e-8 near-duplicate threshold. Existing noise-loss VJP at step6000 is nonzero (.0177220). This is not evidence that arbitrary sample counts recover the true conditional distribution, but it rules out the claim that the recorded candidates are completely identical or noise disconnected.

R2 target-side CCC=.06117063 and one-to-one CCC=.05564621 exceed MAM .04280108/.03693119. Average distinct matched target contents2.5 (MAM2.5625); do not equate these numbers with semantic modes. Duplicate reference slots and their original weights are retained and counted in baseline_per_input.csv.

Raw TRAIN temporal variance AU/VA/expression=.0732767/.0311974/.0404812; speed absolute change=.0214311/.0378090/.0275935. R2 generated values=.0500604/.0176585/.0237993 and .0152893/.0229624/.0124756. Thus lower candidate spread coexists with reduced activity in this descriptive comparison. TRAIN uses deterministic T750 crops and source-weighted sessions; development predictions are full sequence and processed target comparisons are different populations. This is a shape/activity check, not an estimate of true within-input conditional variance.

Decision: test ONE replacement of .25 softmin-cover by .25 one-to-one-cover, with unchanged valid/paired/paired-ES and coefficients. This capacity assignment may distribute target gradients, but cannot guarantee escape from identical outputs or improvement under both quality constraints. Preserve R2-cont as equal-budget control; no additional module, seed, loss weight search or mixed output policy.
