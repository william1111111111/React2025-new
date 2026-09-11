# ES 分项解释（seed123，6000步，T750，K32，两既有bank平均）

| arm | conditional cross | conditional self | group cross | group self | within-input descriptor distance | between-input descriptor distance |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.394347 | 0.308814 | 3.656235 | 2.587544 | 2.443857 | 2.635440 |
| C1 | 0.392616 | 0.293505 | 1.491622 | 1.446551 | 1.391665 | 1.464846 |
| C2 | 0.398461 | 0.308637 | 1.436347 | 1.354873 | 1.256581 | 1.387637 |

ES=cross−0.5self。C2相对C1的条件ES降低，伴随cross增加约0.005845、self增加约0.015132；因此不能将条件ES较低直接表述成每条样本都更贴近唯一paired target。self项本身是评分目标的一部分，这个分解也不意味着“刷分”。

群体ES的降低则伴随cross减少约0.055275、self减少约0.091678。C2在这个有限评估中的within/between绝对距离均小于C1，不能据理论上的额外相似性压力，就宣称训练后C2的跨输入分布一定更分离。理论期望等式不是每个有限minibatch或不同mask下的逐批恒等式。

这些分项须与主报告的任务FRC、目标侧匹配、shuffle-gap和时间形态诊断共同解释。CSV原始精度位于analysis/ES_endpoints_two_bank_mean.csv。
