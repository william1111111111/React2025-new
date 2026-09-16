# 从采样策略 RL 转向生成器级奖励后训练

## 0. 定位与已核对事实

本任务书基于 `william1111111111/React2025-new` 分支 `agent/bert-semantic-controls`，审查提交 `9336013f300dbd31e2fcfbcdecf6d51ba6f3a924`。执行前读取本地实际 HEAD，不能用旧任务书覆盖已经发生的变化。本文件是新实验提案，不是已完成的 REACT 结果。

本次 v2 信号修复已完成：R0/R1 各100更新。覆盖梯度中位数从2.71016e-08变为0.409557；R1质量梯度中位数2.90934，实际每步策略参数delta的中位数0.00154094。固定TRAIN探针输出RMS变化0.000525754。完整联合评价中R1的FRC80=0.937571132、FRD20=130.680369493、S-MSE=0.067711584；父P2分别为0.937901101、130.688674740、0.067720607。无“没有梯度”的证据，覆盖信号数值问题已得到实质修复，但没有形成明显输出分布变化。

当前参数更新只发生在96维低频噪声策略。生成器被冻结，这是旧方案的设计，不是无意断链。停止自动延长该小策略、奖励带宽、KL、学习率的搜索。保留全部旧结果，M0/M1等分支状态保持不变，不擅自恢复或终止其他任务。

## 1. 唯一新主线

实现生成器级、可接受黑盒奖励的在线后训练：借鉴 DiffusionNFT 的正/负隐式向量场目标，让奖励直接决定原有反应生成向量场的参数更新。

基础参考：DiffusionNFT, arXiv:2509.16117，论文 Eq.(5) / Algorithm 1；官方仓库 `NVlabs/DiffusionNFT` 的 `scripts/train_nft_sd3.py`。必须锁定所参考的revision并保留归属。它是既有方法，不宣称算法名称或基础目标原创。这里的REACT组覆盖、质量约束、真实数据replay是任务适配；不照搬原论文的性能保证或速度倍率。

不用下载图像生成器、图像reward models、OCR或Stable Diffusion权重。只移植训练目标与teacher/student管理。不要照抄官方脚本中仅支持LoRA的完整环境。我们的实现是小型REACT向量场的全参数后训练。

## 2. 模型角色和参数所有权

- `ref`：锁定的原生P2，永久冻结。质量参照和向量场正则参照，不成为输出候选，也不与新输出混合。
- `old`：一轮内部冻结的行为生成器。生成候选并提供该轮的旧向量场预测；轮末以 `old = 0.5*old + 0.5*student` 更新。此0.5是首轮固定工程值，不做搜索。
- `student`：P2初始化，解冻 `base.velocity` 的全部8个时序block、条件交互和输出投影。不是只解冻最后一个Linear，也不是只训练 `reward_policy.Policy`。
- P2的speaker ConditionEncoder、BERT特征与现有semantic fusion在本轮固定，所有分支一致；不混入新的语义或prior设计。
- 图中禁止出现新96维输入policy；公开采样仍从原标准局部噪声开始，经训练后的student生成。

新建独立目录 `generator_reward_nft/` 和 `runs/reaction_flow/generator_reward_nft_v1/`。不得修改正被旧实验校验的源码。不要复用会对整个模型执行 `requires_grad_(False)` 的旧 `TrainEnvironment.generator` 作为student。

## 3. 数据与边界

- 使用全部合法TRAIN source人口，不局限于48条语义cohort。有语义时仍用原冻结source-only链，无语义为NULL。
- 继承已有TRAIN paired + 3 alternatives、mask、原始paired时间轴、弱插值规则，第一轮不更改target构造。
- 不读DEV listener、DEV reward来调训练。DEV仅按冻结协议评价。TRAIN块奖励不是官方完整录制FRC/FRD。
- 常量通道、短尾和有效长度按既有合法约定处理；长度<2不计算CCC/DTW，不将无法定义的量当高奖励。
- TRAIN新采样保存 source/crop/target/噪声/behavior-hash 等对应关系。新的noising epsilon必须独立于生成时的初始噪声。

## 4. 生成—评价—再加噪—学习

一轮的顺序固定：
1. 冻结old，抽取16个TRAIN source-block事件，每个用原生Euler16生成K=10条。采样可在 `no_grad` 下执行。保存原生25维轨迹、对应内部clean24维状态，以及所有元数据。
2. 评价每条候选的CCC、精确未平滑块DTW、可观察模式匹配、动态合法性。奖励可以完全在CPU/NumPy中计算。
3. 得到每个候选自己的奖励权重r，而不是十条候选共享一个常数奖励。
4. 对已生成的clean状态重新加独立高斯噪声，重新运行student向量场并反向传播。此处不能no_grad，不能detach student预测。
5. 该轮内old参数不变；训练完轮末再同步old。

保存内部clean状态优于从饱和后的25维输出反变换猜测clean状态；需验证存储的内部状态inverse后确实等于被奖励的原生预测。时间步方向要遵循本仓库约定。

## 5. 核心数学：本仓库采用 noise -> data 时间

设生成的clean坐标为U，独立噪声为eps：

```
u_tau = (1-tau)*eps + tau*U
v_target = U - eps
v_old = old.velocity(u_tau, tau, H).detach()
v_new = student.velocity(u_tau, tau, H)
v_plus  = (1-beta)*v_old + beta*v_new
v_minus = (1+beta)*v_old - beta*v_new
```

第一轮固定beta=1。核心诊断形式：

```
L_core = mean(r * masked_mse(v_plus, v_target)
            + (1-r) * masked_mse(v_minus, v_target))
```

r、U、v_target、v_old均detach；只有v_new拥有参数梯度。不是把negative样本的MSE乘一个负号无界最大化。

在v_new=v_old时，核心形式的输出梯度正比于：

```
2*beta*(2*r-1)*(v_old-v_target)
```

所以r>.5正向学习该生成终点，r<.5提供相反方向，r=.5在旧新相同的瞬间中性。没有reward对Y的导数也可以训练。

正式训练采用官方实践的endpoint自归一化版本，注意时间方向转换：

```
u_hat_plus  = u_tau + (1-tau)*v_plus
u_hat_minus = u_tau + (1-tau)*v_minus
w_plus  = stop_grad(masked_mean(abs(u_hat_plus-U))).clamp_min(1e-5)
w_minus = stop_grad(masked_mean(abs(u_hat_minus-U))).clamp_min(1e-5)
L_nft = mean(r * masked_mean((u_hat_plus-U)^2)/w_plus
            + (1-r) * masked_mean((u_hat_minus-U)^2)/w_minus)
```

mean均按sample有效帧与坐标归一，不让长序列天然有更大权重。tau首轮固定Uniform(.02,.98)，不同时比较时间采样网格。core式用于解析梯度单元测试，正式式用于训练；二者区别写入元数据，不混称完全相同目标。

## 6. 两组奖励设计（本轮不重新展开奖励搜索）

N0-quality：生成器级NFT，使用质量反馈。
N1-quality-coverage：相同NFT、相同质量规则，再加修复后的有效覆盖贡献。

复用v2已经校准并冻结的模式geometry与带宽，保留AU占比较大的已知解释边界，不悄悄改成PCA或删坐标。不能重新用 `u>0` 宣布有效奖励。

每个source组：
- c_k = max_j CCC(candidate_k,target_j)
- d_k = min_j exact_block_DTW(candidate_k,target_j)
- q_k = dual_C*(c_k-mean_c_ref)/scale_C - dual_D*(d_k-mean_d_ref)/scale_D - dual_bad*bad_k
- b_k为修复后的quality-gated coverage leave-one-out边际贡献，按target列先求差再汇总，禁止两个大混合分数相减。
- 原ref独立候选bank只用于TRAIN参考。新ref调用与缓存重用都计入计算成本。

第一次正式轮之前只在固定64个TRAIN校准组上建立每项稳定尺度：质量和coverage分别中心化，使用全校准集std/MAD与固定最小值，不用每组极小std放大噪声。coverage若仍全为0则不给它虚假单位方差信号，定位后停止这项reward而非调DEV。

```
raw_N0 = q
raw_N1 = q + alpha * b
alpha = min(10, max(.1, std_train(q)/max(std_train(b),1e-3)))
centered = raw - mean_group(raw)
r = .5 + .5*clip(centered / Z_train, -1, 1)
```

alpha和Z_train分别预先拟合并冻结；alpha不是事后为通过DEV调的系数，给出两组完整数值。r是训练权重，不是校准概率。

无合法质量支持的候选r上限为.5，不因一组全部表现差而强迫产生“正样本”。所有奖励相同时r=.5；不能把同一个组奖励复制10遍然后标准化，误认为仍有偏好信号。

组覆盖权重依赖其他候选，这是REACT的有限样本近似改编；不声称继承原论文单样本奖励下的严格policy-improvement定理。保留多个合格不同答案，禁止只蒸馏top1，也禁止GT在推理时筛选。

## 7. 保留质量的训练工具（不是保证）

总loss：

```
L = L_nft + .25*L_FM_real + .01*masked_mse(v_new, v_ref.detach())
```

- L_FM_real来自独立真实TRAIN反应，使用已验证FM协议；重建路径在TRAIN使用真实目标属于正常监督，不计为正式生成成绩。
- ref正则比较同一noised状态上的向量场，不逐条蒸馏父模型的最终反应，也不将向量场MSE叫exact KL。
- dual在轮末用raw的组质量与独立父ref差异更新；初值1，EMA .95、step .01、clip[0,20]可沿用旧规则。它是近似约束工具，不保证DEV不降。
- 动态保护不能直接沿用最大speed≈2、acceleration≈4并声称自然。正式训练前用TRAIN真实轨迹和父模型原生输出的逐组速度/加速度q95/q99建立冻结参考，记录而不是奖励更大的运动量。动态异常只有验证了对应场景才定义bad，避免把快速真实动作一律罚掉。

## 8. 预算与调度

两组N0/N1从同一原生P2出发，不从R1的小policy checkpoint出发。

- 200轮，每轮16个source组、K10：每arm 32,000条新的候选块；每轮160个生成训练终点。
- 每轮mini-batch8，共20次参数更新，即4000个真实student更新；同一候选每轮训练一次，每次使用独立noising eps/tau。
- 每个student更新同时用4条真实TRAIN数据replay。每arm16000次真实样本访问，记录unique数据与重复访问。
- 全向量场AdamW lr2e-5，weight_decay .01，clip norm1，前100更新warmup。模型组初始优化器统一新建；不把旧policy优化器state装给student。
- 采样旧模型每轮固定，轮末old <- .5 old + .5 student。
- 保存0/500/1000/2000/4000；1000/4000做完整DEV80，4000固定endpoint做exact FRD20，每arm2000pair；中间checkpoint不拼最佳列。
- 若发生显存约束，只改变microbatch/梯度累积并保持有效预算；不偷偷减少K、时间长度或固定中途截止。
- 新旧实验预算不同，因此不能由NFT>旧policy直接宣称算法等算力优势。N1对N0隔离新增coverage反馈。
- BERT、semantic输入、裁剪/target/noise/tau、referencebank规则共享；两组模型演化后生成结果自然不同，不能要求候选hash仍一致。

这些是一次有明确终点的研发预算，不是达到收敛或超越MAM的保证。不要扩增seed、奖励网格或样本筛选温度，不自动push。

## 9. 必须完成的短接入验证，然后直接进入上述正式预算

不再开展另一个多日审计项目。只做：
1. 统计optimizer param names，证明全部8个velocity block和输出层参与，speaker/BERT/ref/old不参与。
2. 用至少两种不同r的真实TRAIN已生成轨迹，检查loss后student有非零梯度和实际delta，ref/old无梯度且hash固定。
3. old=student、r=.5时，NFT核心中性；r=1与r=0的解析符号正确。
4. 同source同noise短程前后vector-field与轨迹确有变化。不是“有变化即成功”，只是确认权重更新进入了输出。
5. detach采样数据以后仍能更新student；不能删除采样no_grad来绕过bug。
6. 完整断点包括student/old/ref身份、opt、dual、round、buffer索引及RNG；不重复计算已完成官方FRD。

附带reference_checks.py的8组CPU检查只提供数学示例，不能替代以上真实接入。

## 10. 主表与验收

主要基准固定原生P2：FRC80 0.937901101、FRD20 130.688674740、S-MSE80 0.067720607。
研发目标：在同一checkpoint和同一种无GT推理策略下，FRC不少于P2、FRD20不高于P2、S-MSE至少1.5倍，即至少0.10158091；进一步向MAM参考0.1572推进，但不能把MAM数值叫真实人类条件方差。

所有原始结果保留，不以筛选门槛删除不通过的行。报告：完整DEV80、exact FRD20的实际完成范围、目标侧CCC、候选质量尾部、分组DC/slow/fast、速度/加速度、seed/成本及覆盖度。不能将“仍高于MAM”称为“没损失当前最高模型性能”。独立确认尚未完成时只称开发实验。

若多样性提高而质量不满足，只能说是新权衡；若质量好但候选范围没变，不宣布解决多样性。若有梯度和明显输出变化却quality-conditioned coverage不升，进入下一次数据/模型决策，不自动把beta/lr继续扫一遍。

## 11. 暂不并行启动的备选

- 可微CCC/SoftDTW/coverage通过完整rollout直接反传（DRaFT式）是有意义的对照，但它不是原版exact DTW黑盒奖励，不能混称。
- 真正Flow-GRPO需要合法SDE转移、时间方向、噪声方差与log-prob；不能将旧确定性Euler的负FM当log-prob。先完成NFT主线，不同时引入新SDE。
- 显式高低频输出分解是另一条结构路线，但不要和NFT第一次同时改；当前目标是先确认全生成器能否利用黑盒多答案奖励。

## Sources and scope

本地审查：`reward_signal_v2/train.py`, `reward_policy/policy.py`, `mode_supervision/plans.py`, `reward_signal_v2/RESULTS.md`, `reward_signal_v2/JOINT_EVALUATION.md`，均固定审查SHA。
外部基础：DiffusionNFT arXiv:2509.16117，官方NVlabs/DiffusionNFT；DRaFT arXiv:2309.17400；Flow-GRPO arXiv:2505.05470。
本提案未运行REACT训练、未下载新大模型权重、未修改远端仓库。继承数据许可与媒体不外传约束。
