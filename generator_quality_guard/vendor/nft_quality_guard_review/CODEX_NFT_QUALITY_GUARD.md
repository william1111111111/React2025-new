# 保留生成器级 NFT，加入固定参考的任务质量保护

## 0. 任务和证据范围

本任务基于只读审查 commit `bee7d6cb93e9e2b7653bf1df3d170fe78d464b9d`，不是已实施的修改。执行时先读取本地实际最新结果，不覆盖或重复已完成训练/FRD。不自动 push、不新增付费 API、不解冻 BERT、不恢复多 seed，不重新进入 96D 噪声策略。

目标：保留 NFT 改变生成分布的能力，在固定 P2 的质量预算内提高有效候选差异。FRC/FRD 不能靠相互抵消取得合格；没有可接受的分布变化时，不能把近恒等输出称为联合成功。

已核实 DEV80（P2 原生语义配置；NFT 第4000步）：
- P2: FRC .9379011006006148; FRD20 130.6886747400594; S-MSE .06772060692310333。
- N0: FRC .7133259149684557; FRD20 142.77694946679023; S-MSE .13135240972042084。
- N1: FRC .6210042344316955; FRD20 141.67268079636636; S-MSE .13122063875198364。

另一个完整 local TEST 表使用 NFT 第1000步、所有语义 NULL、1142 个双向 source，并计算每模型114200个完整 DTW 配对。它不是4000步，也不是DEV80/FRD20。其3426份特征文件与VAL逐字节相同，不能作为独立隐藏测试确认。保留其全部人口和方向分表，不为新方案使用 local TEST 标签训练或选奖励。

## 1. 当前实现中的确切问题

查看 `generator_reward_nft/rewards.py`：q先与固定P2分数作差，但 weights() 又对每组 raw 减去其均值。于是连续质量排名中的 P2 常数被消掉。绝对参考仍通过支持门槛和dual存在；不能写成“全部参考被删掉”，也不能说所有RL基线化都错。

查看 `train.py`：实际student损失是NFT + .25 real FM + .01同状态ref场MSE。没有对当前student从纯噪声生成的最终轨迹直接反传FRC/距离保护。参考场MSE不是精确KL，real FM不是保住任务分数的充分条件。old每轮向student更新，而固定ref不变；不能把old误用作质量下限。

同一质量向量共同变差，旧组内排名可以不变。正奖励不等于比固定父模型更好。N0也增加候选差异，所以“全部增长由覆盖奖励贡献”尚无证据。真实FM重放、NFT正负权重和其他共有更新都可能参与，需要保留非NFT对照作后续归因，不应据此立即换主干。

## 2. 保持不变

固定原始P2的全部权重/特征变换/语义融合/条件路径为ref，重新从P2初始化两个student。不要从已明显改变质量的N1-4000继续作为主实验。N0/N1检查点保留为分析资产。

保持完整velocity可训练、BERT/condition/semantic分支冻结；保留Euler16、原先独立标准噪声、现有坐标变换、K10正式输出和官方后处理。新推理不读取目标、不作GT选择、不混合P2和NFT输出。

仍保留旧的mask/有效帧处理、原始配对时间轴、显式标为弱监督的same-session替代目标。第一轮不同时改变TRAIN人口、目标插值、AU阈值或结构。

## 3. 质量的三个层级必须区分

(1) 训练块：C=每个候选max_j CCC后对K平均；D=每候选min_j三组加权原生DTW后对K平均。它们不是整条录制的FRC/FRD。官方量在相同人口上是按候选求和，固定K时不要混用平均与求和。

(2) 完整TRAIN监控录制：在共享、冻结的TRAIN目标池/Processor/噪声上计算完整录制分数。用于工程accept/reject，不是独立验证集。

(3) 正式DEV80/FRD20，以及需要时完整原版人口：仍是正式方法评价。不能用(1)或(2)替代；不允许使用DEV/TEST结果在内循环回传梯度或选择候选。

所有质量下限始终相对固定P2；不能每轮相对较差的新old重设预算。第一版默认delta_C=delta_D=0（只有数值等价测试确定的浮点容差）。任何科学/工程容差必须在运行前登记，不随DEV结果放宽。允许未通过而报告，不强行使任务总能推进。

## 4. 奖励层：排名与绝对可行性分开

保留经过TRAIN校准的几何、带宽、候选边际覆盖贡献，但不要继续把固定参考偏差中心化掉作为唯一质量反馈。

- 保存每candidate c_i,d_i和固定ref C0,D0；q_abs=lambda_C*(c_i-C0)/s_C-lambda_D*(d_i-D0)/s_D。
- 同时保存v_C=C0-delta_C-mean(c)、v_D=mean(d)-D0-delta_D；这两个数不中心化、不互相抵消。
- 两项群体约束均未越界时，合格候选的真实模式新增覆盖可得到正反馈；覆盖归因规则保持不变。
- 约束越界时关闭该组的正向覆盖加成，使用相对固定ref的质量反馈及第5节的实际输出修复。不得只因候选是“本组最好”就强制给r>0.5；也不必将整组的所有候选统一判成错误。
- 质量反馈映射到r时用TRAIN冻结的尺度和有界单调变换（首版 .5+.5*tanh(q_abs/Z_Q)）；不做每组减均值。覆盖单独用冻结尺度，不把极小分数按当前组方差无限放大。
- invalid或原支持门槛不通过的候选仍不允许获得正向覆盖；保留质量项的必要负反馈。r是训练权重不是正确概率。
- 固定记录all-below-reference但仍r>.5的比率、两项绝对违例、coverage有效候选比例；任何合成反例中所有候选两项都低于父模型时，均不能因排序获得正权重。

这层只改善反馈定义，不宣称硬性锁住任务分数。

## 5. 学生纯噪声输出上的可微质量保护

新增 `rollout_student_with_grad`。不能调用带 @torch.no_grad 的 `Models.generate()`，不能detach当前student的输出，也不能拿含真实终点信息的 U_tau 估计代替公开采样。16步rollout对student保留梯度，使用checkpointing及候选microbatch；固定P2用相同源/目标/噪声无梯度计算。

第一版每次优化加入一个TRAIN source、K=4、T<=750的quality rollout。两实验组共享该schedule，和NFT/replay schedule独立。该K4是训练估计，不是正式K10成绩。

分别构造（尺度只用TRAIN固定）：
  h_C = relu((C0-delta_C-C_student)/s_C)^2
  h_D = relu((D_student-D0-delta_D)/s_D)^2
  L = L_NFT + .25 L_FM_real + .01 L_ref_field + mu_C h_C + mu_D h_D

任何一项触线就单独产生修复方向；一项改善不能抹去另一项违例。不能把这些写成逐轨迹P2蒸馏；新候选可以靠近不同真实目标。

mu从统一TRAIN校准或预先固定值开始，单独记录两项直接梯度与NFT/realFM/ref梯度。若用dual控制，按真实student输出与固定P2的违例更新，而不是只在组内排名中改变c/d权重。记录控制上限，触及上限不继续无限扩大，应判当前方案难以满足约束。

### 距离梯度：优先使用原生硬DTW的活动路径

这不是把numpy reward强行求导，也不需要把FRD重新命名为Soft-DTW。
1. 对当前student的真实有效帧，使用固定版原生无约束 `dtw_path` 得到每个candidate/target/属性组的最优路径。
2. 路径索引可由detached数组计算。将索引取回torch，在仍带梯度的prediction上重算 `norm(pred[i]-target[j])`。
3. AU/15、VA、expression/8三组各有自身路径；先加三组代价，再对目标选一个整体最小者。禁止各组独立选不同目标后相加。
4. 对固定最优路径与目标支路，前向数值在当前点应等于原DTW；唯一最优路径且距离非零处是普通局部梯度，路径切换/并列处不是全局光滑保证。每次新输出重算路径，不能跨优化步固定旧路径。
5. 零距离用有限的零次梯度；可用torch.linalg.vector_norm，不随意加大eps改变指标。测试非等长、mask、常量、并列和有限差分。
6. 在几条真实TRAIN案例上验证前向数值与冻结官方实现的三组数值一致。附件仅提供小规模DP用于合成测试，禁止将其Python双循环作为750帧生产实现。
7. 若原生路径反向预算无法满足，另行登记Soft-DTW代理版本，不能静默退回32点插值并称为exact。正式验收始终使用完整原生DTW。

CCC同样做torch实现与冻结官方实现的数值一致性测试（常量/eps/通道平均/候选选择）；不是简单以Pearson替代CCC。

## 6. 完整TRAIN质量监控与事务验收

块级指标不能锁住录制级指标。单独冻结16条TRAIN录制，按session和长度做覆盖，不根据DEV分数筛选。冻结完整目标池和Processor结果，固定K10噪声，并缓存固定P2参考。这是训练控制数据，不得称独立holdout。

每100个proposed optimizer steps暂存一次完整事务，在这16条录制上检查：完整FRC不低于固定P2、完整FRD不高于固定P2。记录候选质量尾部、逐source差值和native motion；第一轮尾部只报告，不增加未经登记的多约束。

若验收不通过：
- 不刷新old为该模型，不将其标为locked checkpoint。
- 回滚student、optimizer状态及相应old/buffer模型依赖到上次接受点；保存拒绝checkpoint和全部日志。
- dual/步长控制的变化按预先登记规则记录为控制状态，不悄悄回滚丢失违例证据。
- 下一段使用新的TRAIN提案与噪声编号；不得反复挑固定噪声直到通过。
- 运行以最大proposed budget为上限，不是“直到得到1000接受步”为止。
- 连续3个事务拒绝则终止该arm并明确报告；不得把停止在P2称为多样性提升成功。

这只能保证所接受模型通过指定有限TRAIN监控的数值检查，不能保证未见人口或所有随机采样都不退化。正式DEV检查仍需独立执行。

计算资源：16录制*100候选目标配对/检查*10检查=每arm最多16000条完整DTW配对，另计生成与质量反向成本；记录真实耗时。避免每100步重跑全1142 source。

## 7. 仅两组主实验，固定1000提案步

- Q0-quality-guarded：NFT质量反馈 + 相同实际输出双质量保护 + realFM/ref + 同样TRAIN事务验收。
- Q1-coverage-guarded：与Q0完全相同，新增合格覆盖反馈。

同一P2、seed123、相同初始化和schedule，最多1000 proposed optimizer steps/arm。保持原velocity LR2e-5与warmup规则为起始固定值；如使用预设接受控制减小步长，两arm适用同一规则并报告。不是已知最优超参数。

NFT行为模型刷新/缓冲按原结构组织，覆盖和质量rank修复两arm共有。新开目录，不编辑当前冻结旧实验文件。

预算：每20提案步一轮，16 source*K10=160个采样端点，一轮20更新，每次8端点；50轮共8000 action端点/arm。real replay保持每更新4条。额外质量rollout每更新K4共4000条/arm；TRAIN整段事务监控独立记账。被拒绝、重采样、预热、参考生成都计入成本。

正式最终表报告：proposed/accepted/rejected步数、最后提案模型与最后接受模型（不可混在一行）、FRC80、exactFRD20、S-MSE/FRVar、目标侧覆盖、DC/slow/fast与通道分组。重点直接比较Q1-Q0，同时与固定P2比较。1000提案步不是承诺收敛。

联合研发目标：相对P2无质量损失且S-MSE>=1.5倍（约.10158），并检查新增变化非异常抖动。没有达到时保留实际前沿，不放宽标准或扩大噪声凑数字。

后续要归因“NFT是否优于共有FM重放”，再加入同保护/同预算的NFT权重0对照；当前N0/N1对比不隔离NFT本身。不要用原论文保证替代本任务实证。

## 8. 必要检查和停止条件

先完成很小的真实TRAIN接入检查，再进入正式预算，不无限延长审计：
- 同批学生/父模型的纯噪声rollout、合法mask、不同GT配对选择；
- NFT非零且quality rollout对student确有梯度；ref、old、BERT冻结；
- 绝对质量差变大时保护损失变大，不被组内中心化抵消；
- CCC和硬DTW活动路径数值与冻结指标一致；
- 回滚恢复optimizer/old/RNG状态的等价性；
- 最新日志区分真正的参数变化、质量变化和候选差异。

不改变官方target/Processor/输出域，不从TEST调奖励，不用输出混合当作方法成绩，不自动push。

## 来源和附件范围

源码固定引用：
- https://github.com/william1111111111/React2025-new/blob/bee7d6cb93e9e2b7653bf1df3d170fe78d464b9d/generator_reward_nft/rewards.py
- 同commit下 generator_reward_nft/train.py、objective.py、model.py、runs/reaction_flow/generator_reward_nft_v1/JOINT_RESULTS.json
- 同commit下 runs/reaction_flow/generator_reward_nft_full_test_v1/COMPLETE_RESULTS.md

基础技术参考（不是新贡献或REACT保证）：
- Constrained Policy Optimization: https://proceedings.mlr.press/v70/achiam17a.html
- DiffusionNFT: https://arxiv.org/abs/2509.16117
- tslearn backend/DTW gradients: https://tslearn.readthedocs.io/en/stable/backend.html
- dtw_path: https://tslearn.readthedocs.io/en/latest/gen_modules/metrics/tslearn.metrics.dtw_path.html

`verify_quality_guard.py`仅使用合成CPU张量，验证组内中心化丢失绝对差、质量保护方向、neutral NFT不阻止FM更新、活动DTW路径前向与有限差分梯度、固定参考不逐轮下调等。没有下载或运行REACT模型；不作为真实训练与性能复现。
