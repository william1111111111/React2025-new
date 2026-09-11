# T750首段：seed123实际结果

固定2000步端点，C2的多目标与单配对FRC均高于C0/C1；1000步时C1多目标FRC略高于C2。只有一个训练seed，不能称跨seed一致性或统计显著。后续按已冻结6000步总预算继续，不选最好checkpoint提前停止。

| arm | 多目标FRC | 单配对FRC | 目标侧匹配均值 | 一对一均值 |
|---|---:|---:|---:|---:|
| C0 | 0.455798 | 0.330228 | 0.022755 | 0.019874 |
| C1 | 0.472200 | 0.347461 | 0.021267 | 0.018086 |
| C2 | 0.612683 | 0.433621 | 0.021927 | 0.020694 |

三个arm均actual_steps=training_rows=2000；总预算6000尚未完成，summary.completed=false符合阶段暂停。训练数据、初始化、source/crop/reference/noise调度保持一致。C0有效帧曝光5720135，tensor曝光6000000；其他arm使用相同调度。

C0墙钟645.55秒，C1 988.19秒，C2 1012.75秒。C1/C2部分并行并共享其他GPU作业，不能据墙钟差直接解释方法计算复杂度。峰值约2.34–2.38GiB。

全部12个固定checkpoint、原始逐input/target/candidate结果保存在../task_multitarget。CPU exact FRD与GPU续训现由parallel_budget_v2同时运行；完成情况以其monitor_latest.json为准。T750 held-out分布ES及跨seed最终汇总仍待补齐，不用训练batch ES替代。

Development80为反复使用开发集。MAM只作原生不同预算参考。没有新增lambda/noise bank，未运行独立confirmation。旧报告保持原始准备阶段记录，本文件补充实际首段结果。
