# Mam-Reactor GT-Oracle Candidate-Pool Results

## Scope and provenance

- Checkpoint: `/public/zhaowenjie/react2025_new/mam_reactor/checkpoints/mam_reactor_offline_evidence_epoch0003.pth`
- Checkpoint SHA-256: `ff7c642a3e5aa9ab2effb27fc009498464d040321cbce41a3eef8726837b8553`
- Current-main P4 identity: MATCH
- Split: `B_fit`; contexts: `24`; seed: `1234`
- Scope: train/B_fit GT-oracle feasibility probe; not deployable and not a held-out test score
- GT construction: train-side B_fit files only; ten same-session listener GTs selected with the official test-style rule and aligned to the full speaker sequence by the official target Processor.
- These train-side pilot scores are not the archived 1,142-sample full-test scores and are not directly comparable to them.

## Main result

- FRC: `1.067957` → `1.070601` (Δ `+0.002644`)
- exact FRD: `154.199764` → `153.973807` (Δ `-0.225957`)
- FRDiv: `0.154293` → `0.161375` (Δ `+0.007082`)
- Positive-FRDiv contexts: `24/24` (`100.0%`)
- Median / max FRDiv gain: `+0.006838` / `+0.013007`

## Selected non-baseline variants

- `dynamic_x1.20`: 94
- `style_x1.25`: 31
- `gate_const_0.75`: 24
- `style_x0.75`: 18
- `dynamic_x0.80`: 18
- `q9_uniform`: 1

## Slot replacement counts

- q1: 19
- q2: 20
- q3: 20
- q4: 21
- q5: 22
- q6: 20
- q7: 19
- q8: 22
- q9: 23

## Search operations

- one-swap: 64
- two-swap: 94

## Per-context oracle headroom

| clip | T | pool | ΔFRDiv | ΔFRC | ΔFRD |
|---|---:|---:|---:|---:|---:|
| `listener/session7/Camera-2024-07-19-151039-150947` | 1621 | 55 | +0.013007 | +0.000286 | -0.258182 |
| `speaker/session4/Camera-2024-07-19-143058-143036` | 3334 | 55 | +0.011392 | +0.009635 | -0.028857 |
| `speaker/session13/Camera-2024-06-28-164549-164628` | 688 | 55 | +0.010952 | +0.009606 | -0.022181 |
| `speaker/session15/Camera-2024-07-08-151240-151227` | 418 | 55 | +0.010517 | +0.005151 | -0.056119 |
| `listener/session3/Camera-2024-10-22-123739-123750` | 1917 | 55 | +0.009245 | +0.000734 | -0.310971 |
| `speaker/session17/Camera-2024-07-11-111326-111141` | 1383 | 55 | +0.008660 | +0.000879 | -0.174261 |
| `listener/session12/Camera-2024-07-29-143941-143856` | 873 | 55 | +0.008311 | +0.000573 | -0.164895 |
| `listener/session15/Camera-2024-06-28-164549-164628` | 1262 | 55 | +0.007870 | +0.000222 | -0.310138 |
| `listener/session4/Camera-2024-07-29-160525-160426` | 2574 | 55 | +0.007465 | +0.003667 | -0.155101 |
| `listener/session15/Camera-2024-07-10-134814-134720` | 1080 | 55 | +0.007342 | +0.000389 | -0.288799 |
| `listener/session7/Camera-2024-07-16-143113-143143` | 1590 | 55 | +0.007285 | +0.000615 | -0.093593 |
| `listener/session17/Camera-2024-06-28-120152-120208` | 362 | 55 | +0.007036 | +0.002063 | -0.402782 |
| `speaker/session13/Camera-2024-10-22-173423-173423` | 1653 | 55 | +0.006641 | +0.007107 | -0.201354 |
| `speaker/session4/Camera-2024-07-29-150955-150932` | 6393 | 55 | +0.006339 | +0.000582 | -0.821416 |
| `speaker/session4/Camera-2024-07-11-120318-120232` | 2639 | 55 | +0.006308 | +0.001818 | -0.424758 |
| `listener/session4/Camera-2024-10-15-111253-111325` | 8580 | 55 | +0.006074 | +0.000082 | -0.146778 |
| `speaker/session12/Camera-2024-07-24-114427-114339` | 1829 | 55 | +0.005874 | +0.000420 | -0.401967 |
| `speaker/session17/Camera-2024-07-11-134401-134311` | 1172 | 55 | +0.005796 | +0.006916 | -0.009984 |
| `speaker/session12/Camera-2024-07-22-132508-132549` | 2967 | 55 | +0.005394 | +0.006675 | -0.062528 |
| `listener/session7/Camera-2024-07-24-110700-110644` | 2554 | 55 | +0.005358 | +0.000860 | -0.007552 |
| `listener/session5/Camera-2024-07-19-143058-143036` | 1648 | 55 | +0.004806 | +0.000211 | -0.185488 |
| `speaker/session15/Camera-2024-07-09-111741-111806` | 482 | 55 | +0.003762 | +0.001325 | -0.769536 |
| `speaker/session0/Camera-2024-07-17-112836-112807` | 897 | 55 | +0.003034 | +0.002374 | -0.012894 |
| `speaker/session0/Camera-2024-07-30-123234-123305` | 1591 | 55 | +0.001492 | +0.001267 | -0.112844 |
