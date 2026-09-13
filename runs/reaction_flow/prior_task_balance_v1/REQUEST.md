# 共享先验之后的任务优化平衡：一次有限、等预算干预

## 0. 基准、目标与范围

审查基准：`william1111111111/React2025-new`，分支 `agent/reaction-flow-conditional-law`，提交 `999f073454159183f5941acb8ce1d2cf4949a8dc`。
先检查本地 HEAD、工作区和更晚结果；已完成的改动不重复实施，不覆盖用户修改，不 push，不操作无关训练进程。

用户目标仍是：同一 checkpoint、同一输出策略，FRC 高于 MAM、exact FRD 低于 MAM，同时获得高且有效的候选多样性。不得以 loss、梯度比或 DC 单独替代此目标。

最新提交只做已有输出的 CPU 分析和 sampler auxiliary replay 修复，没有新模型结果。
原生参考：

| 模型 | FRC80 | exact FRD20 | S-MSE80 |
|---|---:|---:|---:|
| MAM | 0.810962319 | 172.576423473 | 0.157203704 |
| T0 | 0.859622576 | 133.876667407 | 0.081215642 |
| G0-local | 0.977341087 | 139.140884363 | 0.061500423 |
| G1-shared | 0.678053116 | 131.143595554 | 0.094315603 |

所有指标属于既有、反复使用的开发协议；FRD20 不是完整 Development80 FRD。MAM 为原生 AU rounding、不同训练预算的系统参考。保留 T0/G0，不将 G1 升为当前最佳系统。

## 1. 本轮只检验一个假设

在同一个 G1 状态和相同 rho=.05 下，维持任务与 FM 在共享参数上的可比较梯度量级，是否有助于恢复任务质量并保留有价值的全局反应变化？

依据：最新已有21个记录节点中，lambda_task * ||g_task|| / ||g_FM|| 的均值 G0=1.55554、G1=.44139；中位数为1.35595/.48180。固定损失系数没有维持原来的实际梯度比例。
这是优化线索，不是已证明的原因；此前只改先验的实验依然有效，不能称其结果无效或不公平。梯度范数不等于 AdamW 实际更新贡献，方向、动量与二阶矩也参与更新。

不新增模型、条件投影、状态编码器、分散度奖励、动态损失、query、assignment、teacher、噪声温度或 rho 搜索。不修改官方指标和生产后处理。不再重复上一轮的完整诊断。

## 2. 对照与共同初始化

两组都从已完成的 **G1-shared step16000** 启动：
`runs/reaction_flow/shared_noise_v1/G1-shared/attempt_000/checkpoints/step_016000.pt`
已读 checkpoint SHA256：
`5a8dc94daf43530b0f304be38e74d27e384c4eac23b1d346e48ff3179c31e792`

路径以实际本地文件为准；验证 hash，不能用另一个模型静默替换。T0/G0 只作已有系统参考，本轮不重训。

- B0-fixed：原 FM + lambda_old * task，lambda_old=1.1338064670562744。
- B1-ratio：相同 FM 和 task，只改变 lambda 的计算日程，见下一节。

共同要求：seed123，各新增2000实际优化步；T750/B4/K_train4/K_eval10；rho=.05；相同源/目标/crop/tau/local/global噪声。
继承相同 G1 模型和 AdamW 状态，LR=2e-5，weight_decay=.01，speaker encoder 冻结；其余优化器配置、坐标变换、mask、损失各内部系数沿用原实现。不要重置其中一组的动量。
训练数据仅来自 TRAIN。新建共有的16001..18000调度，或使用明确记录的新阶段独立 RNG 流；不得循环复用14001..16000。先冻结调度再运行两组。正常继续训练中的重复样本是允许的，不宣称独立新数据。

## 3. B1 梯度比例控制：简单、可恢复，不做搜索

这是一项工程干预，不是完整 GradNorm 算法，也不是论文创新声明。

使用所有 `model.velocity` 可训练参数，对同一次训练 batch 的：
- L_f = 原 FM loss
- L_q = 原完整 task loss（valid + .25 cover + .25 paired）
分别求梯度。L_q 必须来自真实 pure-noise Euler16 rollout，不能来自带 target 信息的中间轨迹。

测量：a=||grad(L_f)||_2，b=||grad(L_q)||_2。
明确 r_target=1.0；这是预先固定的试验值，不代表理论最优，不使用开发成绩定它。

1. 用共有新阶段的首16个 TRAIN batches 在冻结父模型上测量 a/b；不更新模型。初始化两个 EMA 为这16个 a、b 的中位数，保存校准原始记录。校准额外 forward 是工程成本，不算训练步；之后这16个 batches 仍可作为正式调度开头，明确披露。
2. 每20个实际训练步，使用当前 batch 更新 EMA：A <- .9 A + .1 a，B <- .9 B + .1 b。
3. 下一步的目标权重：lambda_goal = r_target * A / max(B,1e-12)。
4. 预先固定的安全范围：[.25*lambda_old, 4*lambda_old]；记录是否触及边界。不可根据 DEV 结果改变范围。
5. 前200个新增步从 lambda_old 线性过渡到当前 clipped lambda_goal，之后使用该 goal；测量间隔内沿用最近目标值。
6. 所有 EMA、比例、lambda 和 norm 均 detach，不对权重控制器反向传播；不计算二阶导数，不额外重归一化最终梯度。
7. B0 在相同测量节点记录 a、b、cosine，以控制诊断计算开销，但始终使用 lambda_old。

保护空梯度、非有限值和完全无任务梯度；异常按实际技术问题记录，不静默跳过更新或继续扩大 lambda。不要把梯度比达到1当成成功验收。

记录每个诊断节点：FM/task loss、未加权与加权梯度范数、r_actual、cosine、当前及下一步 lambda、EMA、cap hit、总梯度范数、优化器参数更新范数与阶段步数。

若比例仍没有进入预定目标量级、或长期 hit cap，必须报告干预执行边界，而不是宣称已经平衡。不得无限追加训练来追某个比例或指标阈值。

## 4. Checkpoint 和必要测试

新命名空间例如 `runs/reaction_flow/prior_task_balance_v1`；历史文件只读。
checkpoint 保存父 hash、两个 EMA、lambda 当前值/目标值、计数器、下一次测量节点、优化器和随机状态、新调度与已消费前缀 hash。
记录分阶段步数0..2000与累计步16000..18000，禁止只修改总步数字段。

必要的短程验证即可：
- B0 初始化输出与父 G1 相同，B1 初始化模型权重也完全相同。
- 两组初始模型/optimizer hash 一致，rho=.05且相同。
- 同一 step 的两组噪声及数据键一致，global跨全部750帧块保持一致。
- B1 的短程连续训练与断点恢复得到相同模型、optimizer、EMA、lambda和调度进度。
- parent/condition 不收到梯度；loss与所有有效梯度有限。
- 已修复的 aux 用local_noise/global_noise可重放，initial_state不再次混入sample。

不要重建已经通过的 evaluator、训练数据契约和整个测试框架。训练前做一次磁盘容量检查，不自动删除历史资产。

## 5. 固定评价和停止规则

保存新增500/1000/2000 checkpoint；两组都按相同固定节点做原生FRC80、S-MSE80、FRVar80和已有DC/slow/fast汇总。中间点用于曲线，不事后替换为另一种选择规则。
只对两个最终新增2000点计算已有exact FRD20协议；不重算MAM、T0、G0、G1已完成的FRD。
保持FP64 Euler16生成、原生连续AU、K10、相同评估基础噪声、目标Processor、完整有效长度。共享先验和两组噪声法则相同。
不得按GT挑噪声或候选，不混合模型，不把某模型的FRD移入另一模型行，不把代理SDTW写成FRD。

报告三个层次：
1. 是否实际提高了B1的任务/FM梯度比，相对B0如何；这只是操作检验。
2. B1相对B0是否改善FRC、FRD，同时保留DC/局部动态与真实目标覆盖。
3. 相对已存T0/G0/MAM，是否形成更有价值的联合点；主目标仍是同一个checkpoint达到质量与有效多样性的联合要求。

最低任务门槛：FRC80 > .810962319、FRD20 < 172.576423473；尽量保留当前约131~140的距离水平。S-MSE>原生T0 .081215642可作为继续投入的一个线索，但不是总体完成标准。MAM的.157203704并未因为本轮改变而被下调。
如果FRC恢复但有效候选差异又接近G0，不把它称为解决；如果DC上升但目标对应变差，也不算解决。
没有新的独立确认集，不作统计等效、多种子稳定性或全面SOTA声明。必要的不确定性用source/session级别，不把800候选当800独立人物。

## 6. 最终只作一个决定

提交实际端点联合表、固定检查点曲线、加权梯度比曲线、分解后多样性及有限说明。
若B1在质量与有效多样性上形成更优组合，保留该控制作为训练配置，再研究贡献组织；不要把通用梯度平衡直接包装成新方法。
若只是在同一质量-多样性曲线上移动，终止这一优化假设，不继续lambda/rho扫描。随后才考虑显式条件整体状态或配对时间路径；本任务中不要实施两者。

本文件是新实验提案，不含已训练或保证达标的结论。

## 阅读依据

仓库同commit：
- runs/reaction_flow/shared_noise_review_v1/REPORT.md
- runs/reaction_flow/shared_noise_review_v1/analyze.py
- runs/reaction_flow/shared_noise_review_v1/gradient_summary.csv
- runs/reaction_flow/shared_noise_review_v1/ccc_attribution_summary.csv
- reaction_flow/shared_model.py
- reaction_flow/train_shared.py

背景工具：GradNorm, ICML 2018（PMLR 80, Chen et al.），用于说明按梯度量级平衡已有研究基础；本任务的简单EMA ratio control不是复现其完整算法。
