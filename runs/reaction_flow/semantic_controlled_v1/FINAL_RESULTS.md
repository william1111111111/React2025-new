# Completed controlled semantic experiment

All three arms completed 500 matched updates from T0 step 14000 and source-only DEV80 evaluation. Exact FRD20 completed all 2000 pairs per arm; queue finished without extension.

| Arm | FRC80 | exact FRD20 | FRVar | S-MSE | temporal S-MSE | TLCC |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.884093 | 132.586720 | 0.058885 | 0.084774 | 0.075483 | 46.625000 |
| E_text | 0.882676 | 131.722047 | 0.055320 | 0.078723 | 0.069984 | 46.474998 |
| E_event | 0.863245 | 131.288260 | 0.053462 | 0.078649 | 0.069816 | 46.487499 |

E0 uses NULL semantics; E-text adds source text/time; E-event also adds event types. Same parent, matched 48-source TRAIN cohort, 500 updates, noise and evaluation policy; DEV has 48/80 non-NULL sources. Official target pool and processors are unchanged.

E-text changes FRC by -0.001416 and FRD by -0.864673 (-0.65%). E-event changes FRC by -0.020848 and FRD by -1.298460 (-0.98%). Both reduce FRVar and S-MSE diversity statistics. Lower S-MSE here is not an improvement in reconstruction error. These results do not show simultaneous gains in conditional association, trajectory distance and diversity.

Limits: one seed, short continuation, small accepted TRAIN cohort, uncalibrated auto-weak labels. Text encoding is a learned byte-mean branch rather than a pretrained language encoder. TRAIN teacher-derived quote proposals and DEV punctuation/word-chunk proposals differ despite the frozen event classifier and role/time gates. No route-level rejection or superiority claim is justified. FINAL_RESULTS.json records exact values, deltas and artifact hashes. Earlier SNAPSHOT.md remains a historical progress snapshot.
