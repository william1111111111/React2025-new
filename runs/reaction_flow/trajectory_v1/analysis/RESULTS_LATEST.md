# Direct conditional trajectory flow — actual status

Development80, native continuous AU, K10, Euler16. FRD20 is the fixed20-source subset, not a full-population score.

| Model | Flow updates | NFE | FRC80 | FRC20 | exactFRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAM native archive | different budget | n/a | .810962319 | .766297889 | 172.576423473 | .157203704 | .058911592 |
| R2 parent6000 | n/a | n/a | 1.181468685 | 1.134674340 | 133.409074259 | .043679938 | .039064668 |
| A | 2000 | 16 | 0.360733618 | 0.357597459 | pending | 0.106513001 | 0.059495833 |
| A | 6000 | 16 | 0.373638849 | 0.391013116 | pending | 0.140510276 | 0.071861960 |

Every flow row additionally inherits6000 R2 training updates through frozen stems/encoder, not its response head. No MAM weights or predictions are used in training.

No source-only task result is claimed from coordinate roundtrip, FM training loss, or noisy-target reconstruction. No new solver/noise-bank/seed/threshold search.

Phase B is prepared but starts only after A12000 sampling diagnostics are inspected; if A produces only noise, inspect the path/masks/integration before task adaptation.
