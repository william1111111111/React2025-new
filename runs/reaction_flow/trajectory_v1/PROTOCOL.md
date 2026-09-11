# Direct conditional trajectory Flow Matching — seed123

This is a new experiment, not a result claim. Branch agent/reaction-flow-conditional-law,
starting HEAD211383b03b388293bf2b231f6a9b3d66052ff9c7. Historical staged endpoints were
already complete locally: S0/S1/S2 FRC80 1.221969005/1.202744548/1.195617702,
exactFRD20 133.312706288/135.929274782/133.822531509. Those assets remain read-only.

## Architecture and representation

Only stems/encoder are copied from this project's R2-parent6000 (hash in manifest).
No R2 response decoder/base/head is loaded. No MAM weights or generated labels.
Warmstart cost:6000 prior project updates,17135366 valid source frames. New field:9,098,520
trainable parameters; frozen speaker modules3,681,574 parameters. New model total12,780,094.
Direct T×24 state, d256/eight self-attention+source-cross-attention+FFN blocks,8heads,
FF1024/dropout0, sinusoidal frame positions distinct from flow-time embedding and
per-block additive time shift. Output projection small nonzero std.01. No codec,
latent bottleneck, fixed candidate identity, bounded residual, repulsion or set assignment.

AU logit and VA atanh clip only to numerical interior eps1e-4. Expression uses a fixed
8×7 orthonormal contrast basis and clip_min(eps) followed by normalization. This boundary-only
choice (rather than adding epsilon at all internal points) preserves float64 interior roundtrip.
It changes exact boundary targets slightly; original target files and metric inputs are unchanged.
Coordinate mean/std use first1024 TRAIN schedule batches/4096 uniformly selected endpoints,
equal occurrence weights and per-occurrence valid-frame means. Fixed std floor1e-3.
Quantiles, domain checks, every fitted crop/target/slot, and roundtrip changes are in coordinate_stats.json.
No DEV/TEST statistics. This fit samples the declared weak distribution, not all human conditional probabilities.

## Data and randomness

One complete14000-record schedule is generated, not a recycled6000-record prefix loop.
Source population and random source-only crop law match R2. TRAIN four-slot pools retain original
paired absolute crop +3 same-session alternatives resized to full source length before same crop.
Uniform slot choice uses an independent fixed RNG, never model costs or scores. Duplicate slot
weights are retained. Gaussian temporal noise and tau use separate keyed per-update streams.
B4 independent source occurrences; one selected real endpoint per source for FM. Loss averages
valid coordinates within each sample, then across samples; target and source masks remain distinct.
R2 schedule construction uses seed123, so repeated TRAIN exposures relative to its warmstart are
possible and expected; they are not a new independent population.

## Training and budgets

A: random new velocity field, frozen condition encoder,12000 actual FM updates,AdamW1e-4/.01,
FP32,T750,B4. Save/evaluate2000/6000/12000, final comparison12000. No dev-driven extension or solver search.
FM draws U,epsilon,tau independently according to the frozen streams; u_tau=(1-tau)epsilon+tau U,
target velocity U-epsilon. No dispersion lower bound, teacher trajectory MSE, quality hinge or task loss in A.

B implementation is prepared: F-cont/F-task from the same A12000 checkpoint and same AdamW state,
2000 new updates, fixed subsequent schedule12001..14000. Encoder remains frozen in both.
F-task retains FM and adds actual pure-noise Euler16 K4 rollout with block gradient checkpointing.
Task=valid+.25cover+.25paired-best, original CCC+calibrated32-grid SDTW cost; no extra paired ES.
Task lambda uses TRAIN16 continuation batches only, median .1*FM-gradient/task-gradient over shared
velocity parameters. No DEV calibration and no trajectory teacher.
A12000 pure-noise activity/quality must be inspected before B starts. A technical success or low
FM loss alone does not automatically justify task adaptation if generated trajectories remain noise.
Current queue runs the complete frozen A budget and its evaluations; it does not silently launch B.

## Sampling and evaluation

Public sample accepts speaker tensors, source lengths, K and explicit [B,K,T,24] noise only.
Euler16 is the fixed first production solver. One frozen TRAIN probe compares16/32; it is not a solver
sweep or a source-only quality result. K1/4/10/32 and candidate chunk tests are structural, productionK10.
Frame offsets refer to absolute source crop start; R2 condition encoder retains its original per-block positions.
Full-recording noise is assembled from independent keyed(sourceID,sample index,absolute750-block start)
Gaussian blocks. No750-block repeats of the same matrix. Candidate identity and common-frame prefixes
are preserved for length changes and source-shuffle comparisons; no equivalence to old32D global-z noise.
Hard750-frame windows preserve the old task source/target/Processor protocol. Time-window context
changes are explicit and are not claimed equivalent to full global-context generation.

Development80 and frozen ten targets/Processor only. Frozen legacy metrics/normalization verified before
cache access or generation. Same native continuous AU across flow checkpoints. MAM remains archived
native rounded-AU reference with different training budget. No hidden-test/participant-independence claim.
Checkpoint+code+coordinate stats+target assets+noise protocol+solver form evaluation identity.
Final checkpoints additionally use one fixed cyclic within-session source derangement, with identical
common-frame noise and a common paired-score mask; no GT choice of permutation.
All25 channels/all10 candidates are plotted for fixed source indices0/1/2, raw and8-frame means.
Grouped raw speed/acceleration and centered diversity accompany source/target-side CCC.

ExactFRD20 uses original unrestricted3-group DTW on the same2000 pairs per final model. No proxy/partial
mean is substituted. Per final point at most four7200-second CPU invocations, bounded8h; progress resumes
from valid pair caches. Parent/MAM metrics are reused, not recomputed. Full80source exactFRD not authorized here.

## Verification status

Initial CPU transform/mask/K/no-target tests passed; full T750/B4 two-update smoke finite.
Exact2vs1+1 resume has zero parameter error and identical optimizer/RNG/numerical loss+draw rows.
TRAIN Euler16/32 MAE.0001093122,max.0022438765 at the two-update model; not a trained-quality claim.
K10 candidate chunk max error5.96e-7. Production remains Euler16.
A full T750/B4/K4/16-step differentiable rollout has finite nonzero velocity/noise gradients, no encoder
gradient, peak allocated memory3.73GB and5.16s technical probe time, without any optimizer update.

No new seeds, no automatic push, no history deletion, no stopping unrelated jobs.
