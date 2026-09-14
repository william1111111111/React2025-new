# Controlled semantic experiment snapshot

Captured 2026-09-14T04:57:01.367214+00:00. All three arms completed 500 matched updates and DEV80 evaluation. Exact FRD20 status below is a snapshot; incomplete scores are not reported.

| Arm | FRC80 | FRVar | S-MSE | Exact FRD20 | FRD pairs |
|---|---:|---:|---:|---:|---:|
| E0 | 0.884093 | 0.058885 | 0.084774 | 132.586720 | 2000/2000 |
| E_text | 0.882676 | 0.055320 | 0.078723 | pending | 1872/2000 |
| E_event | 0.863245 | 0.053462 | 0.078649 | pending | 0/2000 |

E-text FRC is close to E0; E-event FRC is lower in this short single-seed run. Full trajectory-quality comparison requires completed FRD for all arms. Protocol, target pool and source-only inference are unchanged. DEV has 48/80 non-NULL semantic sources. Runtime files may continue changing locally; model checkpoints and exported arrays remain local.
