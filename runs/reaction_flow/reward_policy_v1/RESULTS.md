# Explicit sampling policy results

| Model | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| fixed P2 | 0.937901 | 130.688675 | 0.067721 | 0.055246 |
| R-quality | 0.938003 | 130.687397 | 0.067708 | 0.055232 |
| R-distance | 0.938142 | 130.684883 | 0.067819 | 0.055295 |
| R-coverage | 0.937984 | 130.687299 | 0.067705 | 0.055229 |

Same frozen generator, K10 source-only policy sampling. No target-selected inference; exact FRD uses the fixed 20-source subset. TRAIN block rewards are not official full-record metrics. Single-seed development experiment.
