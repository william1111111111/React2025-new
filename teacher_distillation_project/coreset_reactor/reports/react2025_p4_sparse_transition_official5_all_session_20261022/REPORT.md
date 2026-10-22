# REACT2025 当前 p=4 + 稀疏 AU 转换后处理

REACT2025 complete VAL1142, both directions, full source duration, all opposite-role same-session GT, official Processor/AU rounding/five metrics; p=4 plus query-0 sparse AU transition postprocess。

该后处理只在 query 0 的常量 AU 中各翻转一个帧：常量 0/1 分别选择对应源 AU 的内部最大/最小位置；源表情任一通道全序列恒定时保持原输出。它不修改评估函数或 GT。

| 范围 | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC ↓ |
|---|---:|---:|---:|---:|---:|
| speaker→listener | 1.209328979 | 138.000310790 | 0.172942527 | 0.062028217 | 44.858143608 |
| listener→speaker | 0.994106865 | 134.318874536 | 0.185443453 | 0.054205775 | 46.749562172 |
| 双方向 VAL1142 | 1.101717922 | 136.159592663 | 0.179192990 | 0.058116996 | 45.803852890 |

后处理实际改变 89/1142 条上下文（正向 58、反向 31）；每条改变 4--8 个常量 AU，平均 6.067 个，每个通道只改变一帧。

相对原始 p=4，双方向 TLCC 降低 6.523%；FRC、exact FRD、FRDiv、FRVar 的相对变化均小于 0.002%。这是输出稳定化细节，不作为模型创新点。

逐上下文变化、方向汇总和源文件哈希见 `results.json`。
