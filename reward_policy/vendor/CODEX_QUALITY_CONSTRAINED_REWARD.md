# 有效反应覆盖的奖励学习：保留高质量生成器，先训练采样策略

## 0. 本轮定位与已核实状态

这是研发提案，不是已实施的 REACT/RL 实验。不得将附件的合成数学测试写成模型性能证据。

审查基准：`william1111111111/React2025-new`，分支 `agent/bert-semantic-controls`，提交 `77c231d127c8ea75d570ac42a2b476e1e8ab6a9d`，Git 时间 2026-09-15 04:19:59 UTC。

已完成的 BERT 1000 步结果，来自 `runs/reaction_flow/bert_semantic_v1/RESULTS.md`：

| arm | FRC80 | exact FRD20 | S-MSE80 | FRVar80 |
|---|---:|---:|---:|---:|
| P0-null | .937152 | 130.986103 | .069084 | .056325 |
| P1-byte | .912151 | 129.571334 | .068478 | .053893 |
| P2-bert | .937901 | 130.688675 | .067721 | .055246 |
| P3-bert-time | .932038 | 130.241970 | .068285 | .053571 |

M0/M1 是另一项从 T0 开始、1660 个 TRAIN source、每组 6000 decoder 步的实验。提交快照的共同评价点是新增 1000 步：M0 FRC=.903622、S-MSE=.061120；M1 FRC=.883704、S-MSE=.057296。心跳中两组约到 1556/1553 步。不是最终结果，也没有本轮完整 FRD20。读取本地更新成果，不以该快照覆盖已完成结果，不修改其正在执行的冻结协议。

默认本轮奖励试验使用已完整评价的 P2-bert step1000 作为生成器 G_ref；运行前从实际 checkpoint sidecar 锁定 SHA 和完整指标。如果用户指定另一份高质量 checkpoint，以指定者为唯一父模型，重新锁定协议。不要自动把未完成的 M1 换成质量父模型，也不要将不同模型的最佳列拼接为基准。

核心目标：相对同一个父模型，显著增加有效候选差异，同时保住 FRC/FRD。超过 MAM 和不降低当前父模型性能是不同的标准。

## 1. 为什么先做采样策略 RL

现有模型不是单纯点预测 SFT：它已有 FM 分布学习，以及随机 rollout 上的 CCC/Soft-DTW 任务项。SFT/FM 能学习多个答案；RL 也可能集中到一个高分答案。实验应检验“质量门槛下的组级覆盖奖励”而不是预设 RL 优于监督学习。

`PlanPrior.sample()` 和现有反应 Flow 都使用确定性 Euler 更新。随机初始噪声使最终输出随机，但给定初始噪声后的 ODE 转移不是可直接套 Gaussian log_prob 的随机动作。禁止用负 FM loss 当 log-likelihood，禁止把 velocity tensor 当成 PPO log_prob。

第一版将随机潜变量 z 作为动作，冻结整个高质量生成器。学习显式密度策略 q_psi(z|X)。给定动作的 Flow 生成过程是固定环境。这样可以用不可微的质量门槛、目标匹配和离散覆盖打分，不需要反传完整生成链。

这是真正的单步 contextual-bandit 策略梯度，不要求先实现长时序 PPO/GRPO。奖励和目标只在 TRAIN 更新中访问；正式推理只读取 speaker 和随机数。

## 2. 默认动作空间：原噪声的低频系数；保持 step0 输出一致

这一版不依赖未完成的 PlanPrior。对于一条完整 source 的有效时刻 t，建立常数加三个低频 cosine 的正交基 B[N,r]，r=min(4,N)，用实际 PTS、固定符号 QR，满足 B^T B=I。不要包含 padding；记录短序列退化 mask。

对原模型的每个独立候选、每个变换坐标（通常24维）：

1. 原样生成原始完整录制噪声 epsilon0[N,24]。
2. z0=B^T epsilon0，通常为4×24=96维。
3. z=T_psi(z0, context(X))。
4. epsilon_psi=epsilon0+B(z-z0)。

T 初始为恒等映射。因此 epsilon_psi 与 epsilon0 在初始化时逐项相同，而不是仅有相同分布。保留原 candidate key、全录制噪声、750帧切块、合法 source-only BERT/cache、原 solver 和后处理。

冻结的生成器使用 epsilon_psi 生成反应。策略从输出奖励学习低频随机变量怎样选择；这不保证输出的低频模式与输入低频系数语义一一对应。

同一候选只采一次全录制 z，不在每个750帧块重新抽取；跨块使用完整噪声的对应切片。不同候选独立采样，无固定候选编号 embedding。不要给最终 AU/VA/expression 加随机偏置。

对高斯 epsilon0，z0 与正交残差独立。初始 q0=N(0,I)；训练后 q_psi 改变的是输出边际分布，不是只改变十个候选之间的相关性。因此不能沿用“边际不变，所以质量自动不变”的论证。

备选（不与默认版同时实现）：若 M1 后续完成且有质量证据，同一显式策略可作用于其 plan flow 的初始噪声，冻结 PlanPrior 与 decoder。不要未经单独协议把两种动作空间混跑。

## 3. 策略实现

使用4个交替 mask 的条件 affine-coupling 层；输入通常96维，condition来自冻结 source encoder/BERT 的已允许上下文汇总；conditioning MLP两层宽128。scale使用有界log-scale，例如 .25*tanh(raw_scale)。最后预测 scale/shift 的线性层置零，实现严格恒等初始化。不要令整个 conditioner 永久不参与梯度。

必须实现：
- sample/rsample_and_log_prob；
- inverse 和 log_prob(z_detached|context)；
- 高精度 change-of-variables logdet；
- masked dimensions/padding 排除；
- q0等于标准高斯的测试。

log probability 是动作 z 的真实联合密度。求和所有有效动作维度；禁止用维度平均伪称联合概率。分布更新中的 z 必须 detach，不能把采样路径误当成 REINFORCE 的 score。

公开采样：X -> q_psi(z|X) -> frozen G_ref -> K=10 输出。禁止先生成几十条、再依据 GT/官方分数选十条，禁止在测试调用奖励器。

冻结 base encoder、BERT、semantic fusion、Flow velocity、输出变换和统计量。冻结权重不能保证质量不变，因为动作分布仍然改变。

## 4. 奖励只读取 TRAIN；先确定能评分的实际范围

默认一个训练 episode 是完整 speaker context + 一个预先选定的真实750帧生成块，K=10候选，沿用现有 TRAIN paired+3 alternatives 的合法目标构造。选择 block 与动作无关，保留真实有效长度，不重定 paired 时轴。完整录制 z 在各 episode 中仍按完整长度定义。

这是块级训练奖励，不是完整 DEV80 指标。不得把块级 exact DTW 称为 full-record official FRD；所有最终主表必须用完整录制官方协议。若改成全录制奖励，应新冻结预算，不能静默改变。

参考模型在同一 TRAIN source/target/block/K 下产生基线分数。参考随机流与训练动作分离；所有 RL arms 共用同一冻结参考表和阈值。不得从 DEV/MAM 分数调训练阈值。

单候选定义：
- c_k=max_j CCC25(Yhat_k,Y_j)；
- d_k=min_j D_DTW(Yhat_k,Y_j)，组顺序和权重沿用官方实现；
- a_k 表示有限值、输出域、极端速度/加速度等筛查结果。

在第一版中，对实际750帧有效长度计算非平滑 DTW，不通过32点采样冒充 exact。标注为 TRAIN-block metric。若成本无法满足固定预算，记录真实成本并重新提出版本；不要偷偷使用代理然后仍声称直接优化了 exact FRD。

第一轮不引入 LLM 奖励模型或学习型 critic。已存在数值真值可以直接打分。没有可靠依据的心理/人格标签不进入奖励。

## 5. 质量门槛 + 有限的目标支持覆盖

独立 TRAIN calibration 选择64个分层 source-block、每个父模型K=10，记录所有分数和来源。阈值仅作为未校准的工程门槛：c_floor用每个校准群组父模型best-target CCC的低分位；d_ceiling用其DTW高分位。正式用到新episode时，先计算该episode的独立父模型参考，以相同规则确定阈值。不可用模型当前输出选择最有利目标阈值。

尺度 s_C、s_D、s_Phi 使用 TRAIN calibration 的固定稳健尺度。状态描述 Phi 复用 mode_supervision/plans.py 的可观察低频量或其分组均值/方差；在实际预测属性上提取，不能只看策略动作有多分散。

针对每个候选k、真实参考j计算：

u_kj = 1[CCC_kj >= c_floor AND DTW_kj <= d_ceiling AND a_k=valid]
       * exp(-||Phi(Yhat_k)-Phi(Y_j)||_normalized^2 / 2).

同一对(k,j)同时通过质量条件；不要分别借用两个不同目标的好分数来宣称覆盖了该目标。官方 FRC/FRD 各自选最佳目标的报告规则保持不变，以上只是更严格的训练覆盖定义。

D(S)=sum_j w_j max_k u_kj，初始w_j=1/J；没有候选时最大值定义为0。

性质：
- 多一份同样的好答案通常没有新增覆盖；
- 在另一真实参考附近新增好答案可以有新增覆盖；
- 远离所有真实反应的异常值没有覆盖奖励；
- D在[0,1]，不无限最大化S-MSE。

这度量的是有限、弱适当参考池支持的覆盖，不是完整人类条件分布或语义模式数。不把所有same-session变化都要求每个输入等概率复制。第一版不自动扩大目标池，也不逐输入聚类后硬要求每类一个候选。

## 6. 将“不降低质量”作为两个独立约束

优化 E[D(S)]，同时控制：
E[mean c] >= E[mean c_ref] - delta_C；
E[mean d] <= E[mean d_ref] + delta_D；
坏候选比例不高于父模型的相应参考范围。

默认工作目标delta_C=delta_D=0。有限采样下的代理约束和KL都不是泛化或逐输入质量保证。最终开发表必须重新验证。

训练回报可写为：
R(S)=D(S) + lambda_C*(mean c - mean c_ref)/s_C
            + lambda_D*(mean d_ref - mean d)/s_D
            - lambda_bad*(bad_rate-bad_ref).

使用正的、独立的dual系数；当CCC约束不满足时增大lambda_C，当距离约束不满足时增大lambda_D。初值1，dual步长.01、范围[0,20]、历史batch滑动平均；这些是待冻结的第一版工程配置，不是理论最优。

动作分布KL正则到q0=N(0,I)，初始beta=.01。记录总action KL及每维诊断，勿混淆二者。策略越界时早停本次新试验并保存真实状态，不无限放宽KL或质量约束以追多样性。训练中的质量目标与多样性门槛分开，避免一个大权重总分掩盖FRC下降。

## 7. 组级奖励的 credit assignment：第一轮不用直接抄 GRPO

候选动作给定 X 独立采样。完整组回报 R(S) 可以直接用于
  grad E[R] = E[R(S) * sum_k grad log q(z_k|X)]。

降低方差时使用 counterfactual/difference baseline：
  A_k = R(S) - R(S without candidate k)。

删除候选后的回报只用其他候选；质量项继续以原K归一，参考阈值与目标权重不变。它对z_k独立，因此on-policy的期望score梯度仍正确。不要将K-1伪装成官方评价；这只是baseline计算。

绝对不要将同一组的相同R复制给K个candidate，再做组内减均值；这样组奖励会全部变成0。也不要不加检验地把difference advantages再组内居中，破坏已经设计好的credit assignment。

默认每批新rollout只做一次on-policy REINFORCE更新：
  loss_actor = -mean_episodes(sum_k stopgrad(A_k) * log_prob(z_k.detach()|X))。

若需要尺度控制，只使用过去batch的固定RMS来整体缩放，第一轮不做候选间相对排序或硬winner-take-all。

KL用独立reparameterized policy samples进行pathwise估计：
  z_kl, logq_kl = policy.rsample_and_log_prob(X)
  KL = mean(logq_kl - logN(z_kl))。

区别：reward动作detach以获得score梯度；KL样本不detach以获得正确的KL梯度。不能把对旧样本的简单mean(logq_new-logq_ref)误当成完整KL梯度。

第一轮不做多epoch PPO，因此不用近似组联合importance ratios。若后续增加PPO，必须记录真实old joint action logprob；逐候选clipped surrogate只是局部近似，不能声称多轮更新精确优化组级分布。

## 8. 有限实验，不再无限审计或权重搜索

先用16个预固定TRAIN source-block、每个32次父模型采样做一次可达性诊断。用参考标签判断合格候选中是否存在更多真实低频差异。可展示GT辅助的组选择作为诊断，但不能进入正式主表。这个有限样本池既不证明理论上界，也不能证明没有其他模式。

之后三组同架构、同初始化、同source/target/base-epsilon预算，均只训练小策略：
1. R-quality：只有质量反馈（D=0）；
2. R-distance：共同质量约束 + 有界/归一化pairwise-distance bonus；
3. R-coverage：共同质量约束 + 上述质量门控真实支持覆盖。

固定原始采样器作为不训练参照，训练compute=0单独报告，不假称等算力。

默认seed123；B=2 episode/iteration、K=10；500次on-policy更新/arm，共10000候选rollout/arm。学习率1e-5，AdamW weight_decay=0，clip_grad_norm=1。预定100/500检查点，只以500作为本阶段终点；实际CPU距离开销和GPU调用量单独记录。没有候选池推理选优，不扩rho或reward权重网格。

这些是提案配置，正式调用本地代码前一次性写PROTOCOL并锁定。若可达性很低或likelihood契约测试失败，先报告，不把小采样策略强行训练成万能解法。

不占用已承诺给正在运行M实验的资源、不自动终止其他进程、不修改其冻结文件。不自动push、不覆盖历史产物、不访问付费服务。

## 9. 验收与后续决定

同一个策略checkpoint+同一个固定G_ref+同一种公开采样设置，K=10，无GT选择，完整DEV80：FRC80、S-MSE80、FRVar80、目标侧CCC、分组DC/slow/fast；固定FRD20全部2000配对完成后报告，仍明确不是全80 FRD。

比较候选平均质量与尾部质量，防止多数相同高分答案掩盖少数异常值。奖励用的状态描述和外部诊断都保留，不能只在reward feature上报告胜利。

工作目标：S-MSE >= 1.5 × 父模型，同时FRC >= 父模型、FRD20 <= 父模型，继续报告与MAM的独立系统对照。它是研发目标，不是效果承诺；若小容差允许，则在实验前另行写清楚，不事后改口叫零损失。

R-coverage与R-distance检验奖励结构，R-quality检验“RL只追分”会怎样；这组实验不能自动支持“RL优于所有SFT”。若要证明RL算法本身的独特价值，后续增加同样可微奖励的直接反传对照，并匹配采样/生成预算；不要一次把该对照、全decoder RL、SDE采样都混进来。

若冻结生成器的奖励策略取得有价值的联合结果，再考虑PlanPrior策略或很小的decoder adapter。若只移动质量—多样性权衡，不继续权重网格；需要提高生成器可达的好模式能力。全Flow RL可参考Flow-GRPO的有密度随机转移，但ODE->SDE改变必须单独做step0质量和采样一致性检验；不能往当前Euler随意加噪声并声称分布不变。

## 10. 最小必须测试

- 同初始随机种子，identity策略与父模型初始噪声/预测一致。
- affine coupling forward/inverse与logdet；logprob有限且pad不参与。
- 奖励中的重复好答案不增加覆盖，新增合格模式有奖励，异常值无多样性奖励。
- leave-one-out score梯度与可枚举离散toy的精确梯度一致。
- reward_actions detach；policy梯度非零，frozen generator与BERT无更新。
- 不同K的prefix、跨块z一致性、长尾真实mask。
- 奖励绝不读DEV/TEST目标；测试sampler签名没有reward/GT参数。
- 连续更新vs断点恢复保留policy/optimizer/RNG/dual/KL统计/进度。

## 参考技术，不作为新颖性或性能保证

- Flow Matching for Generative Modeling，arXiv:2210.02747。
- Training Diffusion Models with Reinforcement Learning（DDPO），arXiv:2305.13301。
- Directly Fine-Tuning Diffusion Models on Differentiable Rewards（DRaFT），arXiv:2309.17400。
- Flow-GRPO: Training Flow Matching Models via Online RL，arXiv:2505.05470。
- KL-Regularized Reinforcement Learning is Designed to Mode Collapse，arXiv:2510.20817。
- Constrained Policy Optimization，ICML 2017，PMLR 70。
- Density estimation using Real NVP，arXiv:1605.08803。

现有方法已经涉及分布学习、集合覆盖、奖励后训练。不能将“RL”“无Hungarian”或“有可学习采样器”单独当成足够的新贡献。最终定位应由真实任务联合收益、模式有效性及明确比较支持。
