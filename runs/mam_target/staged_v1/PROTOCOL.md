# R2 quality-protected staged fine-tuning, seed123

This is a development experiment, not an established diversity improvement.
All arms warm-start R2-6000 d9b0e07aad91f3bbb254bfc2ad5780f9fa2b9a138245dbafd0c79fa935bf8936.
The parent exact FRD20 is already complete: 133.40907425912587. It will not be rerun.
Parent FRC80=1.1814686849299616, S-MSE80=.04367993772029877, FRVar80=.039064668118953705.

## Fixed comparison

S0_quality: original R2 + separate CCC/SDTW score guards; all participating parameters except A0 prior.
S1_joint: S0 + ramped six-component dispersion; same parameter policy.
S2_staged: S1 objective; stochastic head/norm and FiLM only through new step500,
then decoder body through2000; encoder/stems/base always frozen.
All three use new AdamW, not inherited momentum; dropout is off for student/reference.
Activity LR: source/base 1e-5, decoder body2e-5, stochastic/FiLM5e-5, wd.01;
all arms multiply LR by .5 from new step1501. No teacher trajectory reconstruction penalty.
Unused conditional prior remains frozen. There are no new network modules.

T750/B4/K4, exactly2000 new updates; shared TRAIN continuation6001..8000 from
refine_v1, distinct from parent training6000prefix. The earlier A/B experiment remains paused.
Reusing that already frozen tail for this separately identified experiment is explicit, not new independent data.
Source/target/crop/noise selection and original R2 weights are unchanged.
Training alternatives are full-source linear resize then source crop, not the evaluation Processor.
No DEV labels enter training or calibration. No other seeds or search variants.

## Auxiliary definition and calibration

Original R2: valid+.25cover+.25paired+.5pairedES; b=.15813484622234084,32-point SDTW surrogate.
Guard: separate relu current-minus-parent per-input quality scores, scaled by TRAIN positive medians.
TRAIN16 calibration fixes s_C,s_D and six positive spread epsilon values (one per group/representation).
State uses 8-frame means, actual tail-frame mean counts as one pooled timestep.
Dynamic subtracts each candidate's pooled temporal mean. VA/2 only in this auxiliary space.
References are deduplicated by file ID for dispersion only; original quality slots retain weights.
Common valid range is min source and target lengths. Fewer than2 unique targets skips components;
fewer than2 pooled points skips dynamic. Offdiagonal mean uses K(K-1), not K².
Alpha=.25 min(u/500,1), lambda=lambda_max min(u/500,1), u starts0.
The two ramps intentionally make the initial auxiliary effect weak. No dynamic controller.
Calibration lambda_max=.1379777975863425 from TRAIN16 only, grad ratio .2 with cap10.

Torch2.1 fused no-grad attention and differentiable execution can differ in FP32.
Training/calibration select the same unfused attention operations via an identity key clone and
fused-dispatch flag; actual activation/weights/dropout are unchanged. Small remaining convolution/
FP32 differences are recorded in calibration (output tolerance2e-6, guard1e-5).
At the common parent point guard and its ReLU(0) gradient are analytically zero, so lambda calibration
uses the original R2 gradient as grad L_Q there, avoiding numerical hinge activation in calibration.
Actual subsequent training evaluates the specified separate guard against no-grad parent scores.
Public inference is the unchanged sampler; numerical export protocol is inherited and separately hashed.

## Evaluation and resource budget

GPU1 only; three arms may run concurrently. Checkpoints saved new500/1000/1500/2000;
task evaluation only500/1000/2000 (total6500/7000/8000). Main comparison final2000.
Development80, frozen1+9 targets/Processor, K10/globalnoise/750block, continuous AU unchanged.
Exact FRD20 only at final2000: all2000 candidate-target pairs per model, original unrestricted DTW.
Each final model gets at most four7200-second resumable CPU invocations (8h total); incomplete counts
are explicit and never produce a full FRD mean. No parent DTW repetition.
GPU monitoring and manifests record actual updates/frames/time/memory and checkpoint hashes.
MAM archive is a native rounded-AU, different-budget system reference, not matched architecture control.

Quality band: FRC80>=1.1223952506834636, FRD20<=140.07952797208216 and both better than MAM.
These are development engineering tolerances, not equivalence or guarantees.
No GT selection/reordering, checkpoint mixing, output mixture, added noise banks or training seeds.
Noise VJP and direct fixed-noise versus negative-noise response are logged every50 steps.
Full raw candidate exports, CCC matrices, target-side/assignment diagnostics, grouped diversity,
raw speed/acceleration and block boundaries remain available. Assignment is analysis only.

## Verification before formal launch

10 unit/regression tests passed (initial and second test logs retained).
Real T750/B4/K4: all three arms complete two finite updates; S2 continuous2 vs1+1 resume
has parameter max error0, identical optimizer/RNG/rows. Frozen weights match parent exactly.
The smoke updates are extra engineering cost and are not counted toward formal2000-step arms.
