# Existing reward signal audit

All numbers below reuse saved TRAIN data; no official FRD rerun.

| Saved population | Groups | Mean coverage (old) | Exactly zero affinities | Tiny positive (<1e-12) |
|---|---:|---:|---:|---:|
| calibration | 64 | 0.0216669 | 42.58% | 36.91% |
| reachability | 16 | 0.0289624 | 34.72% | 45.21% |
| reference_table | 1000 | 0.0245958 | 44.04% | 39.25% |

Old calibration distance contributions: AU99.975%, VA0.0064%, expression0.0186%. Four modes of one physical AU coordinate account for about79%. Old speed/acceleration ceilings are approximately2/4, near legal-range extrema; zero bad_rate is not evidence of realistic motion.

Held-out old quality gates: separate best-target tests pass 81.67% of candidates, while joint same-target tests pass 12.08% of pairs. These denominators differ and the two numbers are not interchangeable.

Bandwidth-only h2=410.192; stratified geometry h2=2.48668. The latter changes geometry and is separately versioned. Held-out stratified qualified distance shares: {'AU': 0.9726249293075723, 'VA': 0.0045159619923178865, 'expression': 0.022859108700109605}.

Raw masks, gates, quantiles, marginal credits and variance summaries are in the JSON artifacts. Missing old policy action matrices/trajectories are explicitly listed there.
