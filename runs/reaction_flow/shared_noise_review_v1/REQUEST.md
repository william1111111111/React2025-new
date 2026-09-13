# 共享噪声实验审查：先定位相关性损失，不自动启动新训练

## 0. 基线与本轮权限

审查基准 commit：979b8a830b1c480d03d868b3883247454dabe077。
仓库：william1111111111/React2025-new。
分支：agent/reaction-flow-conditional-law。

先读取实际 HEAD/status。已有更晚结果时读取它们，不回退或覆盖。
本轮只允许读取已有结果、进行 CPU 统计，以及修复辅助采样返回接口与必要的单元测试。
不启动新 GPU 训练、不开 rho 网格、不新增 seed、不重复已完成的 exact FRD20。
不混合模型输出、不根据 GT 选噪声或候选、不改官方指标、不自动 push。

## 1. 已完成结果，必须完整保留

- T0 parent：FRC80=0.8596225763708395；FRD20=133.87666740682033；S-MSE80=0.08121564239263535。
- G0-local：FRC80=0.977341087330938；FRD20=139.1408843633149；S-MSE80=0.0615004226565361。
- G1-shared：FRC80=0.6780531163513746；FRD20=131.14359555386798；S-MSE80=0.09431560337543488。
- MAM 原生归档：FRC80=0.810962318916204；FRD20=172.57642347297096；S-MSE80=0.15720370411872864。

G0/G1 各完成 2000 新优化步、5716122 有效 endpoint 帧，FRD20 各 2000 个配对完成。
这是 Development-80 / 固定 FRD20 子集的 seed123 结果，不是隐藏测试或多种子结论。
T0 保留为当前较均衡 Flow 参考；G0 保留为较高 FRC 参考；G1 保留为相关先验机制对照，不升级为联合目标已经达成的主模型。

主要证据文件：
- runs/reaction_flow/shared_noise_v1/FINAL_RESULTS.json
- runs/reaction_flow/shared_noise_v1/PROTOCOL.md
- runs/reaction_flow/shared_noise_v1/decomposition.csv
- runs/reaction_flow/shared_noise_v1/block_diagnostics.csv
- runs/reaction_flow/shared_noise_v1/G1-shared_noise_factor_probe.json
- reaction_flow/shared_noise.py
- reaction_flow/shared_model.py
- reaction_flow/train_shared.py
- reaction_flow/sampler.py

## 2. 完成现有结果的紧凑汇总，不新增生成

从 decomposition.csv 按 source 等权汇总，AU/VA/expression 用 15/25、2/25、8/25 加权；
输出 G0/G1 的 total/DC/slow/fast，及三个属性组各自值。与历史 T0/MAM 使用同一代数分解和分母。
验证逐 source/group 的 total=DC+slow+fast，再报告汇总。不得把快速成分直接命名为噪声，或把 DC 直接命名为语义模式。
区分高精度代数 S-MSE 与官方 FP32 实现，原官方表不变。
block covariance 按相同 source/相邻 block 对应比较，明确尾块长度；不要把绝对 covariance 上升当成标准化相关系数上升。
已有共享因子探针只有四条 source，只能支持这四条上的因子敏感性，不推成全数据语义解耦。

## 3. 唯一新增研究诊断：CCC 精确分解

目的：定位 G1 的 FRC 差异主要来自哪一项，而不是马上换网络。
对 T0/G0/G1 的已有完整预测、已有 processed targets 和原始 10x10 CCC 矩阵进行 CPU 分析。

对每条 candidate：
1. 先用原版官方 25 通道 CCC 的规则选择 target j*；不得按通道分别重新选 target。
2. 对这个固定 candidate-target 对，逐通道记录预测/目标时间均值、方差、协方差。
3. 记录 Pearson r、标准差比和均值失配项。
4. 对非退化通道核验恒等式：
   CCC = r * 2*sigma_p*sigma_y / (sigma_p^2+sigma_y^2+(mu_p-mu_y)^2+eps)。
5. 官方常量通道/NaN/epsilon 行为必须读源码遵循。Pearson 未定义时单独标记，不能用错误的零替代后解释形态；官方 CCC 汇总仍保持其原规则。
6. 在每通道重建 CCC 后再按官方方式汇总；不能用“平均 r × 平均校准因子”代替平均 CCC。

分两套配对解释：
A. 各模型原生最佳目标，解释它们实际的官方 FRC。
B. 固定 G0 的最佳目标索引，T0/G0/G1 都与此目标比较，只作匹配固定的诊断，防止目标切换掩盖差异。
两套分表，B 不替换任何官方成绩。

至少返回：每个组的均值失配、标准差比、Pearson r 分布、常量通道比例；按 source 聚合而不是把全部帧或全部候选当独立人群。
可额外计算“将均值失配项设零”的纯代数敏感度，必须标为使用 GT 的诊断，不能生成或发布为可部署结果；不将其命名为保证上界或真实模型性能。

根据结果只作一个下一步判断：
- 均值失配主导且时间形态基本保留：优先研究 TRAIN 真实反应摘要的条件分布，而不是继续加大原始 global noise。
- 形态/相位关联主导：优先研究配对时间条件路径，先不新增 DC 奖励。
- 多项同时变化：明确证据不足以归因，保留结果，不用单一因果故事解释全部 FRC 差异。

## 4. 修复公共采样的辅助返回契约

当前 SharedFlow.sample 将 base_local/base_global 合成为 initial，传给 ReactionFlow.sample；
后者 return_aux['noise'] 实际返回 initial，而不是 base_local。
直接把 aux['noise'] 重新作为 SharedFlow.sample(noise=...) 输入，会再次混合；默认产生的 global 也没有完整返回。
这不影响当前 evaluator 显式提供两组基础噪声的已完成结果，但影响公开接口重放。

将辅助返回字段明确为：
- local_noise：实际基础局部噪声；
- global_noise：实际基础共享噪声，rho0 可为 None；
- initial_state：已组合、mask 后的状态；
- prior_metadata：rho、坐标与键语义；
- predictions、valid_mask、NFE。

旧字段 noise 若保留，必须与公开 noise 参数采用同一 base-local 语义，或明确弃用并报错；禁止静默返回另一种含义。
不要改变生产预测与 checkpoint 参数。

测试：
- 使用返回的 local_noise/global_noise 重放预测一致；
- rho0 与原父接口一致；
- K-prefix、candidate chunk、padding；
- 相同候选的 global 跨 block 不变；
- 初始状态只组合一次；
- 历史分数/预测不因返回字段调整而重算。

## 5. 对先验变化的数学说明

rho=.05 保持单帧方差为1，但时间均值方差比为 1+rho*(T-1)。
T=750 时是38.45，标准差比约6.2008。
因此不能称它为对全部时间模态都很小的5%扰动；这也不是输出均值方差必然增加38.45倍的断言。
固定任务权重并不保证先验改变后 FM/task 的梯度比例不变。使用已有诊断日志比较实际分项梯度，只解释为优化线索，不擅自重新校准后开启训练。

## 6. 返回成果

在独立目录写一份短报告：
1. 原生联合任务表；
2. G0/G1 与 T0 的精确多样性分解；
3. CCC 失配来源及其证据范围；
4. 采样辅助接口补丁和测试；
5. 一个有依据的下一步建模方向，仍标记未实施。

不要把这份任务扩成新的多阶段训练矩阵。当前要避免继续在噪声幅度上反复试验，却没有解释它如何影响相关性。
