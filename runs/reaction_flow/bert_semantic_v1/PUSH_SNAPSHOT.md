# BERT experiment push snapshot

Captured 2026-09-14T12:15:33.143246+00:00. Four concurrent pipelines on GPU7; runtime files can continue changing after this commit.

| Arm | Training step | Stage | FRC at500 | S-MSE at500 |
|---|---:|---|---:|---:|
| P0-null | 980 | training | 0.8870233869119819 | 0.08131420612335205 |
| P1-byte | 864 | training | 0.892512685140914 | 0.08435876667499542 |
| P2-bert | 862 | training | 0.8754657806360922 | 0.08557789027690887 |
| P3-bert-time | 862 | training | 0.8708461949341457 | 0.08691741526126862 |

1000-step full joint evaluation and fixed-intervention diagnostics are complete only when their own result artifacts say so. Midpoint metrics are not final selection evidence. Original model/data/noise/dropout protocols remain frozen. Parallelism changes execution only; see PARALLEL_EXECUTION.json.
