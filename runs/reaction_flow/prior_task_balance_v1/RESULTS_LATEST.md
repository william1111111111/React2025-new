# Prior/task balance — actual progress and endpoints

Single GPU1, seed123. Both arms inherit identical G1-16000 weights/AdamW, rho=.05, 2000 new steps. TRAIN calibration16 + short smoke24 updates are disclosed engineering cost, not production steps. Native continuous AU, FP64 Euler16, Development80/K10; exactFRD20 is the fixed subset, complete2000pairs only. No new generation policy or metric definition.

| System | new steps | FRC80 | exact FRD20 | S-MSE80 | FRVar80 | target coverage |
|---|---:|---:|---:|---:|---:|---:|
| MAM archive | — | .810962319 | 172.576423473 | .157203704 | .058911592 | .042801 |
| T0 archive | — | .859622576 | 133.876667407 | .081215642 | .053756177 | .058758 |
| G0 archive | — | .977341087 | 139.140884363 | .061500423 | .046456564 | .065558 |
| G1 parent | — | .678053116 | 131.143595554 | .094315603 | .039878245 | .050465 |
| B0-fixed | 500 | 0.777425786 | None | 0.084364310 | 0.039635520 | 0.054124 |
| B0-fixed | 1000 | 0.851592574 | None | 0.079776861 | 0.044677597 | 0.055744 |
| B0-fixed | 2000 | 0.789414698 | 127.55576905062028 | 0.068110809 | 0.036372185 | 0.051039 |
| B1-ratio | 500 | 0.902498939 | None | 0.075066142 | 0.044553705 | 0.056617 |
| B1-ratio | 1000 | 0.947270331 | None | 0.062405087 | 0.044539880 | 0.056538 |
| B1-ratio | 2000 | 0.827581122 | 126.19868951795426 | 0.049628735 | 0.031184716 | 0.049141 |

Controller: goal ratio1 is prespecified, not a success metric. Per-node norms/cosine/update norms and cap flags are in gradient_ratio_curve.csv. The [0.5,2] diagnostic band is descriptive only and never changes weights, bounds, budget, or checkpoint choice.

B0-fixed: actual steps=2000; cap fraction=0.000; last500-step median weighted ratio=0.4905974956660367; fraction within[0.5,2]=0.44.

B1-ratio: actual steps=2000; cap fraction=0.000; last500-step median weighted ratio=0.9062551807564525; fraction within[0.5,2]=0.88.

Decision: End this optimization hypothesis after the authorized2000steps: no measured joint improvement meeting the prespecified quality floor and preserving diversity/coverage over B0. Preserve all endpoints; no lambda/rho scan or added budget. Explicit global state or paired-time conditioning remains unimplemented. The MAM S-MSE reference is not reached; the full objective is not declared achieved.

DC/slow/fast use equal-source weighting and group weights15/25,2/25,8/25; algebraic total is separate from official FP32 S-MSE. DC is not a semantic mode count, and fast is not automatically noise. Target coverage uses the actual saved target-side max CCC over full target slots, not800 independent people. Reused development data; MAM output rounding and budget differ. Gradient norm ratios are not AdamW update contributions.
