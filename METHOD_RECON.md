# Reaction Program 方法前置核查

日期：2026-09-17。范围：本轮只完成 STEP 1–5；codec、executor、planner、RL 与 EXP-A/B/C 新运行尚未开始。旧产物、暂停的 Q0/Q1、冻结 RM A/B 均不修改。

## 1. 仓库与代码事实

当前工作区 `/home/zhengshiyi/react2025_new` 是独立 HiRP 及后续实验仓库。用户列出的 `dataset/react_2025.py`、`framework/metrics/*` 实际位于依赖仓库 `/home/zhengshiyi/react2025`；新仓库 README 明确记载这一拆分。数据与 Python 环境是软链接。当前 `discuss/diversity/`、`discuss/sacrt/` 不存在；对应证据可在旧仓库 `discuss/methods/`、`innovation_runs/20260725_sacrt_gate0/` 和本仓库历史报告找到。不能称已读不存在的文件，也不将旧 memory 当作当前执行代码。

核查覆盖数据、指标、Processor、HiRP/TaskHiRP、Flow/BERT、mode supervision、奖励/NFT、旧 reaction_space 离散原型与 SACRT Gate0 的相关实现和结果。并非声称逐字读过所有第三方依赖及每份历史日志。新目录的 RECON_EVIDENCE.json 将保存关键文件哈希。

## 2. 实际训练 GT

- 旧 `dataset.react_2025.ReactionDataset` 的非 test 分支：同文件名、对侧角色、相同 crop 起点的 paired 25D/58D；普通 val 不自动构造十目标测试集。长度补齐的历史行为不能当作真实可观测帧。
- HiRP paired loader 区分 source_lengths 与 pair_lengths；source-only 推理不得使用目标长度决定输入长度。早期 group losses 使用同 session 的分布证据，不等于同步一对多。
- 当前 Flow/T0/P2 路径：`reaction_flow.data.FlowData -> mam_target.data.TaskData`。每个 source 构造 4 槽：paired + 3 个同 session TRAIN listener。后三者完整序列线性插值到 source 总长度，再取相同 crop；代码明确标注 weak-label alignment。
- Flow FM 从固定日程的 4 槽中抽一个 endpoint；任务 rollout 使用集合质量损失。P2 继承 T0，并新增独立 BERT 融合分支；不是纯 paired 训练，也不是天然正确事件对齐。P2 续训只曝光 48 条有可用弱语义的 TRAIN source，1000 实际更新。
- 新路线禁止复用上述 donor 逐帧插值作为 executor 真值。Executor 仅使用真实 paired listener 自己的动作与时间；跨录制只增加可追溯 program support，不提供 raw trajectory endpoint。

## 3. Reference 数量与协议

当前实际 `dataset/react_2025.py` 共 475 行，test 分支 `k_select=9`：paired + 9。同角色对侧 session video pool；去掉 paired 后足够则 random.sample，不足则 random.choices，单例可重复 paired。对原 speaker 方向，目标为 listener；对 reverse 方向，源为原 listener、目标为原 speaker。身份由方向决定，不能一律写 listener。

旧 `DATASET_GT_CONSTRUCTION_AUDIT.md`、README/memory 的 paired+10/full_sequence_eval 记录引用的是历史 >1100 行 loader；当前文件不包含该开关。**历史 11-reference 结果不可混入当前 10-reference 主表。** 此次不篡改旧结果，也不修改官方指标代码。

当前新仓库已冻结的 DEV80 清单为 paired+9、K10；全量 VAL 是 571 原 speaker 方向，已有本地 TEST 清单为 1142 双方向且也是 paired+9。后者原始 source/target 特征审计显示本地 TEST 和 VAL 全部重合，不是独立隐藏测试。

新的统一主清单采用既有双方向 1142 条清单的确切 reference 次序、重复项和随机种子，将路径映射为内容一致的 VAL；逐文件核验哈希后冻结。新模型不得重新抽 reference。另保留原 DEV80 及 FRD20 为历史快速诊断，其 reference/人口与主表分别记录。主表固定 10 个 reference，不能将 11 个直接切成 10 个然后沿用旧分数。

Processor：`framework/modules/post_processor.py`。长度相同时直接使用 GT；长度不同时按 1000 帧分段，调用固定 reaction autoencoder 扩缩时间，AU 阈值化、expression softmax。它不是简单线性插值，内部可能随机采样；处理后的 target 缓存也必须固定哈希、实现、checkpoint、配置与 RNG。只有 reference 文件名固定并不充分。已有处理缓存若被清理，必须恢复并核验或新版本重算后全部模型共同使用，不能声称旧分数自动等价。

## 4. K=10 如何进入指标

官方 FRC：对每个候选，在所有 reference 上取逐通道 CCC 均值的最大值，再对 **10 候选求和**、对 source 求均值。不是候选均值，也不是一一匹配；值可以大于 1。每个候选可命中同一个 reference，故高 FRC 本身不证明覆盖。

官方 FRD：每个候选与每个 reference 分别计算三块 DTW：AU 的 DTW/15 + VA 的 DTW + expression 的 DTW/8。对 reference 取最小，10 候选求和，source 平均。必须计算 native 完整长度 DTW；32 点代理不叫 FRD。

S-MSE：10 候选展平为 T×25 向量，两两平方距离总和除以 K(K−1)T25，再 source 平均；完全不读 GT。候选 DC 均值差异也计入，不等于语义多样性。

Temporal S-MSE：先按每候选/通道移除时间均值，再用同一候选间公式，是附加诊断。FRVar：对时间维 `torch.var`（样本方差默认 correction=1），再候选、通道和 source 平均。单帧序列需明示异常，不擅改官方实现。

三项核心指标直接读取 **25D 输出**。58D 不直接进入 FRC/FRD/S-MSE；它用于 source conditioning、渲染或可选辅助目标。视频画质、planner 准确率、RM 分数不替代三项指标。

## 5. 25D 与 58D 每维的可证实含义

| 0-based 索引 | 已确认的含义 | 未确认部分 |
|---|---|---|
| 0–14（逐个 AU channel 00…14） | 15 个 AU occurrence；已有 TRAIN 实测为二值 | 实际 AU 编号/肌肉顺序未从提取器获得，禁止假装第9列是嘴角 |
| 15–16 | VA 两维，文档顺序为 valence/arousal | 未定位原提取器 schema，保留该来源级别，不能反向用情绪解释“证明”顺序 |
| 17–24（expr 00…07） | 8 类 expression 概率 | 确切类别顺序未核实；不能擅套 AffectNet 常见顺序 |
| 58D 的 0–51（coef 00…51） | FaceVerse 52 个 expression basis coefficients | 不是 52 个已命名 AU/动作；没有可证实的一维一肌肉解释 |
| 52–54 | FaceVerse rotation 三参数 | 相机坐标/轴符号、提取器约定仍需溯源，不直接贴 yaw/pitch/roll 标签 |
| 55–57 | translation 三参数 | 单位及世界坐标解释未核实 |

证据：baseline README 的数据定义、`framework/utils/losses.py` 的分块、FaceVerse `merge_coeffs2(exp,angles,translation)`。机器可读逐维列表另存 CHANNELS.json。形状常为 `[T,1,58]`，只移除此 singleton，不能 squeeze 掉候选轴。

## 6. 最强 baseline：没有三指标同时最强的一个模型

以下为现有已完成记录，不是本轮重评；DEV80 的 FRD 来自固定20-source 子集。

| 模型与 checkpoint | FRC80 ↑ | exact FRD20 ↓ | S-MSE80 ↑ | FRVar80 |
|---|---:|---:|---:|---:|
| MAM 原生归档 | 0.810962319 | 172.576423473 | 0.157203704 | 0.058911592 |
| R2 step6000 | 1.181468685 | 133.409074259 | 0.043679938 | 0.039064668 |
| S0 quality step8000 | 1.221969005 | 133.312706288 | 0.043695860 | 0.040877711 |
| T0 step14000 | 0.859622576 | 133.876667407 | 0.081215642 | 0.053756177 |
| P2 BERT +1000 | 0.937901101 | 130.688674740 | 0.067720607 | 0.055246 |
| B1 ratio +2000 | 0.827581122 | 126.198689518 | 0.049628735 | 0.031184716 |

S0 是这组已核查 DEV80 记录的高 FRC 参考；P2 是近期固定生成器、完整全量评价可追溯的质量参考；MAM 保留多样性参考。不能把 P2 叫作已经全面超过 S0/MAM。原生 MAM 的 AU rounding 与 Flow 的连续 AU、不同训练预算都要保留。

本地 TEST 双方向1142（与 VAL 内容重合）P2：FRC 0.786617945、完整 exact FRD 127.926324864、S-MSE 0.066014679、FRVar 0.053226778、temporal S-MSE 0.056890124。原 speaker 571 子群 FRC 0.988703947 / FRD119.723068871 / S-MSE0.066830314；reverse 子群更低。不能拿原 speaker FRC 配双方向 FRD 拼成一行。

MAM 旧独立报告的 full-test budget0.15 为 FRC0.853953/FRD151.620708/diversity0.152125，但 reference manifest、输出与本轮主协议未完成匹配，故只列为旧证据，不并入统一主表。

EXP-A 将锁定 P2 checkpoint 作近期基线，同时保留 S0/MAM 的协议一致性复评入口；本轮只冻结清单，不冒称已完成统一协议 EXP-A。

## 7. 已尝试的方向和边界

- HiRP group/session score、输出标度和 task refinement 提升部分质量，但有低多样性 tradeoff。
- shared noise/梯度比例：多样性有相当部分来自 DC/整体状态，不足以支持语义模式解释；已有审查不支持继续搜索 rho。
- BERT 四组对照：P2 FRC≈P0，不能声称文本已显著解决多模态反应；时间偏置并无明确联合优势。
- mode_supervision：低频 QR/cosine recording plans，不是 observable compositional action ontology。旧 M0/M1 不作为新 program 的已验证证据。
- NFT N0/N1：增大 S-MSE 同时降低质量；固定质量保护 Q0/Q1 保持暂停。
- RM A/B step500 及 vector selection：B 质量头排序优于双头，但所有候选池门槛不足10而回退，不能据此启动 RL。
- SACRT `76d32c98ca30b7b2` 的 Gate0：retrieval、transport、post-warp mode、endpoint support 全 FAIL，决策 STOP。它反驳该实现的 raw transport 可用性，并非否定所有语义 support 假设。
- 旧 `reaction_space/discrete_code.py` 是冻结 VAE + k-means prototype，不是端到端 RVQ codec，也不是有视觉证据的 action programs。旧弱集合 W2000 在120帧/32例上 FRC0.430929 < P0.471140，多样性增大；打乱输入反而更好，条件关系未建立。
- `discuss/methods/ANCHOR_AND_DIVERSITY_BUDGET_ABLATIONS_20260901.md` 表明 MAM 加预算0.20的收益很小，0.30不再改善，去掉预训练 anchor 严重塌缩。它不能证明本轮应通过扩大噪声刷分。

## 8. 可复用与不可直接复用

可复用：实际 TRAIN/VAL 文件索引与 PTS、哈希与分折审计、source-only typed cache、UNKNOWN 处理、官方 Processor/指标、固定 source/reference manifest、native DTW 已校验实现、padding/mask/K-prefix 和时间边界测试、现有 speaker 转录及未经验证的事件提议。

不可直接复用：same-session raw 时间拉伸监督、GMM/RL/Flow 主结构、RM 当作真值、匿名 AU 强行命名、按字符均摊时间、session/描述性文件名作为模型输入、旧 VAE k-means 直接冒充 RVQ、旧 11-reference 成绩、已复用 audit 当独立测试。

## 9. STEP 3–5 的数据契约与停止点

首批选择600个 **TRAIN event candidates**，按 session 平衡、recording 覆盖、固定 seed/hash 排序，从原 speaker 转录和已有 source-only 提议选取；不读 listener 运动决定 speaker 事件。沿用保守 recording-date 分组；第一批仅 RM_fit，避免复用 RM_audit 参与 ontology。日期分组不等于已验证身份互斥。

时间/WHO 仅在现有证据可追溯时附加，不足时 NULL/UNKNOWN；不能把600 candidates说成600合格标签。保留低活动/unknown，不把缺标解释成 maintain。Listener observation、event equivalence 与 reaction support 单独文件/权限；只有真实动作观测和关联都有证据才写 observed_support。

LLM prompt 先 open vocabulary；动作 ontology v0 为开放格式与归并规则，未得到真实视觉描述前不编造类别频率或“已发现”的动作词表。精细时间从帧PTS/数值轨迹，LLM 引用给定时间锚；不能凭文字生成 AU/3DMM。confidence 是未校准自评。

最新任务书提出后续人工抽检；本轮仅准备分层核验字段，不伪造审核、不要求用户现在提供人工结果。未经核验的自动弱标签与 strict/gold 隔离。没有新的 API 调用或媒体上传。

报告 STEP1–5 后停止，待下一步再实现 codec/executor。EXP-B/C 必须显式 oracle 诊断：来自训练内保留折的 program/timing 或专门隔离的评估器；VAL/TEST listener 不用于 ontology/标注调参，不能把 oracle 输入当正式 source-only。Q1/Q2/Q3 尚未通过。
