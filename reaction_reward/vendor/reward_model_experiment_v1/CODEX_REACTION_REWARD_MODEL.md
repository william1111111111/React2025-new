# REACT 条件奖励模型实验任务书 v1

## 0. 本轮任务与执行边界

目标：独立训练一个输入 speaker 多模态序列 X 与 listener 25D 轨迹 Y、输出条件匹配分数的奖励模型。首先验证裁判是否能够识别输入—反应关联、时间联系和多个合理候选，再考虑其对生成器奖励后训练的价值。

本轮默认只执行阶段 A（奖励模型训练与离线验收）；阶段 B（接入 NFT）是预先设计的后续对照，不自动启动。没有实际训练结果的预告。模型叫“数据支持的条件匹配裁判”，不是经过人类偏好验证的心理/语义正确性裁判。

已读取基线：william1111111111/React2025-new，agent/bert-semantic-controls，bee7d6cb93e9e2b7653bf1df3d170fe78d464b9d。启动前核对本地最新代码，保留历史实验，不自动 push，不上传媒体、不调用付费 API、不新增人工标注要求。不修改正在运行的协议。

## 1. 核心假设与不做的推断

H1：上下文与时序联合输入，比只看 listener 的判别器更能预测有证据的条件对应。
H2：加入弱多反应支持而不惩罚未配对答案，能提高多个合格候选的覆盖，而非只识别录制身份。
H3：加入同一输入下的生成候选偏好，能提高裁判在未用于训练的生成器输出上的排序可靠性。
H4（后续）：通过 H1–H3 验收的冻结裁判，在相同显式质量约束下，帮助 NFT 提高有效多样性。

重要：奖励模型分数提高不等于官方 FRC/FRD 提高；配对不是全部可能的正确答案；相同 session 不等于相同逐帧条件；未发生的答案不等于错误；跨 session 和时间移位也不是天然的语义负例。

## 2. 数据划分：先分真实互动，再裁剪和生成

只使用官方 TRAIN 范围构建本轮 RM_fit / RM_cal / RM_audit，目标比例为70/15/15，seed123。

1. 以可核验的 dyad/原始互动 recording group 为最小分组单位。正反方向、同一次互动的多摄像头、同源裁剪、同源音视频和属性序列必须同折。
2. 有可靠参与者ID时，优先使用参与者或dyad不重叠划分；若连通分量太大，保留不拆分并报告实际比例。只有录制分组可核验时，明确叫 recording-disjoint，不能叫 person-disjoint。
3. 原始文件哈希和来源记录决定重复合并，不用模型推断的人脸身份作为事实。
4. 先划分，再抽选 same-session donor、cross-session donor、错时样本和生成候选。一个比较的 X、Y_A、Y_B、所有 donor 必须均来自所属折。负样本不能绕过划分隔离。
5. RM_fit 训练参数与所有归一化；RM_cal 选择预先列出的checkpoint、拟合temperature和阈值；RM_audit 在冻结选择后只做一次独立裁判验收。
6. 不用 DEV80、VAL 或本地 TEST 的标签训练/调奖励模型。已知本地 TEST 与 VAL 特征重复，不能把二者作为两个独立检验。
7. source encoder 主实验独立初始化，复用输入和代码接口，不加载已由全TRAIN监督训练的 P2 encoder 权重。这样避免把看过 RM_audit 标签的任务编码器当成独立奖励模型的特征来源。
8. P2/NFT 是历史候选生成器，可能训练过全部TRAIN。生成候选在 RM_audit 上的检查只代表“裁判未用这些组训练”的泛化，不代表生成器从未见过这些组。
9. 额外输出按 session 的分层统计；第一轮不要再自动启动多折/多seed。后续证明跨场景泛化需另做预先留出完整session的复制实验。

## 3. 输入与预处理

X：speaker_audio[T,768]、speaker_emotion[T,25]、speaker_3dmm[T,58]，真实长度与PTS。
Y：listener属性[T,25]，真实长度与PTS。

- 使用原始25D合法属性，不给裁判输入target ID、source ID、session ID、生成器名称、GT距离、filename、标注confidence、该候选是否被挑选。
- metadata仅进入采样、划分和评价，不进入网络。
- speaker和listener各有mask。上限750帧；不足长度正常mask，填充值不参与pooling/difference/attention。
- 不用长度归一化、不同补零方式或是否有插值作为标签线索。比较内匹配有效长度或共同物理时间窗；任何统一采样操作对正负两边对称执行。
- 对Y加入原始一阶差分与真实delta-t信息，先局部编码再降采样；不是把高频信息先平均掉。
- 只在RM_fit拟合统计量。既有外部预处理统计的出处需记录。
- 弱同场景轨迹不要冒充逐帧对齐真值。可以用于整体head；时间head保持mask。
- 初始主实验不引入BERT/event type/LLM分数，以避免语义缓存覆盖差异混入奖励实验。需要时后续增加独立BERT消融。

## 4. 五类证据，不是统一的“1与0”

### A. 真实 paired
同一次互动的原始时间轴、同一合法crop，作为观测强证据；通过基本文件/域/遮挡或检测质量标志检查。不是声明每一帧行为都在人类意义上最优。
用于上下文偏好、活动片段上的时间偏好。

### B. same-session alternative
使用同折其他录制的反应，保留原始来源。若只有session成员关系，则只视为弱整体支持，权重0.25；不比较它与原始paired谁更好，不把它作为逐帧同步正例。
对一个X，从弱支持池抽取至多3条。候选相似也正常，不强制情绪各不相同。
当没有合格跨场景对照或证据不足时标UNKNOWN，不制造偏好标签。

### C. 跨场景错配
真实listener片段作为候选，选择同折、长度/活动量/强度接近、场景不同的source构造弱错配。随机错配只提供统计证据，不称“绝对不合理”。
默认证据权重0.5；中性/低活动、内容关系无法区分的样本可以UNKNOWN。
每个Y和X在正/负位置平衡复用，尽量避免单输入边际可预测标签。

### D. 时间错位
同一原始互动内取真实连续片段，使用预定±2秒和±4秒shift的实际PTS索引，裁到共同有效窗口。禁止np.roll、循环拼接、只给负例补零、在负例中插入人为跳点。
仅对具有可测活动/事件变化、shift前后确实非等价的片段建立弱时间偏好（默认权重0.5）；平静反应、合法延迟歧义或有效区域不足时UNKNOWN。活动门槛只由RM_fit确定。
平移前后都落在合理延迟范围不能自动负标；不同shift分别报告，不混成一个准确率。
时间偏好训练s_time，不因为大shift就同步给整体s_context判负。

### E. 生成候选偏好
冻结候选来源，默认P2与N0-quality_step1000用于RM_fit/RM_cal；N1-quality-coverage_step1000只用于冻结后的生成器外推诊断。若本地资源缺失，仅报告缺失，不用另一个checkpoint静默替代。
所有来源使用一致合法条件；首轮统一无文本的speaker特征路径，原生连续AU、标准噪声、固定Euler16。不得把DEV导出的预测搬来训练。

同一X的候选A/B只在两项独立任务证据均占优时产生偏好：
C_A >= C_B + 0.01，且 D_A <= D_B - 0.05*max(D_ref,eps_D)。
C是每候选的最佳目标CCC，不是求和K10的FRC；D是完整有效训练窗口的原生分组DTW，非32点代理。eps_D只用于稳定零距离，来自训练固定尺度。
这是“指标派生偏好”，不是人类偏好或官方全录制分数。
CCC与DTW各有所长、差异不足或证据冲突：UNKNOWN，不是TIE。
生成器名字不决定好坏；至少一半生成比较在同一生成器内部完成，以免只学来源。
两条都合格但不同的候选可用于多答案接受诊断；没有独立等价证据，不强迫它们分数完全相同。

## 5. 小型候选库预算

冻结选定窗口清单后，最多：
- RM_fit 256窗口 × 2生成器 × 16候选 =8192候选窗口；
- RM_cal 64窗口 × 2生成器 ×16 =2048；
- RM_audit 96窗口 ×3生成器 ×16 =4608；
- 合计上限14848候选窗口，不是14848独立录制。

各折按真实互动分层、平衡session和活动量取窗。不足时报告实际数量，不跨折补齐。历史TRAIN缓存只有其split/source/crop/model/noise/condition provenance均匹配才复用。
每个X/来源的候选数固定，不按结果追加直到出现偏好。每窗口至多保留8条有margin的生成偏好，避免少数可排序输入主导。额外保留未用于监督的UNKNOWN对用于检查多解排斥。

## 6. 奖励模型网络

同一骨架用于全部对照：

SourceStem(768/25/58 ->256) -> 2层temporal Transformer
ListenerStem(25 + delta25 + dt1 ->256) -> 2层temporal Transformer
两层双路cross-attention；width256、heads8、FFN1024、dropout0.1。

先使用局部卷积/池化将时间token降到约T/4，mask按真实有效元素处理；添加基于真实秒的相对位置关系。不要把“同帧”当作唯一正确延迟，使用可学习相对时差表示，不固定所有人都零延迟。

输出：
- s_context：整体上下文/反应状态匹配，使用双方全局池化交互；
- s_time：跨序列时间变化关联，使用逐时cross交互再mask-pool；
- 返回两个实数及组合分数，不输出未经校准的“正确概率”。

全局头和时间头分别有任务mask。弱same-session不进入时间损失。
源、反应独立编码器与融合层均训练；不更新P2生成器。

## 7. 偏好损失

对有依据的偏好A>B：
L_rank = w * softplus(-(s(X,A)-s(X,B)))。

- 上下文样本作用s_context。
- 错时样本作用s_time。
- 生成器偏好作用s_context+s_time。
- 真正独立标注为TIE时才用0.5软偏好交叉熵；本轮纯自动管道默认不制造TIE。
- UNKNOWN: weight=0，必须完全不参与梯度，仍保存/计数。
- 每一类先按有效权重归一，再跨类相加，避免大量easy negative淹没少量hard evidence。

默认：L_base_context +0.5 L_time +0.25 L_weak_context +0.5 L_generated。
这些是初始实验配置，不是最优或理论常数。

## 8. 三组主实验

RM-Paired：base_context + time；same-session不被当负例。
RM-Multi：与RM-Paired相同，再加弱same-session整体支持。
RM-Gen：与RM-Multi相同，再加生成候选偏好。

相同网络、初始化、数据划分、base/time采样日程、seed123和4000更新步。每批32条比较记录，目标组成16 base/8 time/4 weak/4 generated，缺类时对应loss mask，不偷偷用更多base替换，实际有效曝光逐类记录。
三组每步的基础样本完全相同；新增监督会改变active computation，不声称严格等FLOP。
AdamW lr1e-4、weight_decay0.01、warmup200、clip1。step1000/2000/4000保留；RM_cal按预定hard-task宏平均pair NLL选择其中一个checkpoint，每类等权、记录覆盖；选择后在RM_audit一次验收。
第一轮不自动多seed、模型集成、超参网格或奖励训练中的在线更新RM。

## 9. 裁判验收：不只看随机负例准确率

必须输出任务分层的pair ranking accuracy、pair NLL、coverage/abstention、按互动分组bootstrap区间：
1. paired vs matched weak cross-scene（统计代理，非人类合理性标准）；
2. 活动片段的correct vs non-circular shifted（时间代理）；
3. 未见生成器的clear Pareto偏好（独立于RM训练样本，但标签仍是指标派生）；
4. same-session多候选接受/拒绝分布（弱支持保留，不预先规定所有alternative必须高分）；
5. 两项指标均合格且模式不同的生成候选能否同时被接受；
6. 全录制score与窗口score一致性，仅作诊断，不把block匹配叫作官方全录制质量。

捷径检查：
- X-only和Y-only在平衡的条件关联任务上另训轻量探针；同X两候选的X-only天然抵消，不把这个简单结果包装为强证据。
- 重复Y与不同X形成平衡2x2诊断，报告 I=s(X1,Y1)+s(X2,Y2)-s(X1,Y2)-s(X2,Y1)，pure source-only或listener-only加性评分在此会抵消。语义不明确的cross组合不强标错误。
- 正/负交换顺序，mask/padding与batch chunk不应改变分数；source/target元数据不得进入forward。
- Y-only在明显损坏轨迹检测上得分高是合理的，不要求所有任务Y-only都chance；要检查主条件关联是否超过它。

启动后续RL的建议门槛（工程筛选，不是公认标准）：
- active temporal pair准确率>=70%；
- unseen-generator Pareto pair准确率>=65%；
- 平衡hard condition任务比Y-only至少高10个百分点；
- 相关主任务的group-bootstrap区间下界>0.5；
- 不以多解大幅拒绝换来以上表现；RM-Multi/Gen的弱多反应保留应不低于RM-Paired；
- 无teacher/test泄漏；按来源统计，生成器身份不能单独解释偏好。
样本不足无法估计时报告inconclusive，不降低阈值凑通过。

## 10. 无需更新生成器的价值测试

在冻结RM_audit池（特别是未见N1生成器）中，对每个X固定16条候选：
- 固定first10/预定random10；
- RM选top10；
- metric-oracle选择仅作使用GT的参考诊断，绝不作为推理结果。

冻结RM的选择只看X/Y，不看候选的官方指标或目标。取完再由独立代码评估CCC、DTW、候选差异与被接纳模式数。若选出的高RM分候选只是在奖励分数上高而真实任务质量无改善，不进入正式RL。
Top10筛选可能收缩差异，所以本测试首先检验质量排序，不声称它本身是高多样性的最终采样器。它是best-of16的诊断，不冒充原生只生成K10的主表，计入16次生成成本。

## 11. 校准与绝对质量

偏好模型的原始score有任意尺度和可依赖X的加性偏移，不直接拿sigmoid(s)>0.8作为“80%正确”。
RM_cal可以拟合head-wise温度，仅校准所定义代理任务上的两候选偏好概率。
以后对RL使用同一个X上的固定P2参考候选得分差：
Delta_s_h = (s_h(X,Y)-mean_k s_h(X,Yref_k))/temperature_h。
这会抵消source-only offset，但不是对真实合理性的概率保证。
source条件不同、score分布不同，不能通过一个未经检验的全局0.5阈值处理。
不在候选组中心化中抹掉固定质量底线；RM参考和官方CCC/DTW参考分开保存。

## 12. 后续阶段B：仅在裁判验收之后

从相同原P2开始、冻结同一已选RM、继续全velocity NFT，BERT与source冻结。三组共享绝对质量保护、原生输出、真实FM重放、参考场约束、同预算：
- G-metric：原数值质量反馈+相同有效覆盖，不使用RM；
- G-RM：加入冻结RM质量反馈，不加多样性bonus；
- G-RM-div：同G-RM再加质量合格后的覆盖bonus。

G-RM-div vs G-metric：学习裁判是否比原手工质量反馈提供帮助；G-RM-div vs G-RM：多样性奖励是否有额外作用。这是系统对照，不是把每一项算法因素都纯隔离。
首轮预算1000提案更新，实际生成成本与ref/gate成本都记账，不因拒绝/回滚无限延长。奖励只读取RM_fit，监控使用TRAIN内分开的RM_cal并明确它已成为训练控制数据；RM_audit不参与自动rollback、奖励标定或采样。
绝对质量约束仍分别对固定P2的CCC和原生距离；不认为“奖励模型更高”就等于FRC/FRD被锁住。冻结RM、Ref并不保证student质量。
目标是同checkpoint的FRC/FRD不劣于共同P2，S-MSE至少1.5倍且有目标支持；不是性能预测。
无GT参与公开推理。公开生成K10，不进行GT/真实listener/RM候选筛选来冒充原生采样。

## 13. 奖励投机防护

阶段A冻结后RM不与生成器同轮在线追逐。检查高RM分而低CCC/高DTW的案例、静态偏置、高频运动、幅度极端化、来源识别。不能只看RM自评分曲线。
RM误差大时优先补可解释标注证据；单纯堆3个同偏见裁判不保证安全。第一轮只训seed123，不默认奖励模型三seed集成。
所有自动偏好主要学习数据与指标代理；要宣称人类对未观察反应的真正偏好，需要独立有效标注，不能由模型自己打分循环证明。

## 14. 工程文件建议

reaction_reward/
  split.py                # 原始互动/重复连通组划分与manifest
  evidence.py             # typed preference + UNKNOWN mask
  negatives.py            # role-safe同折采样、非循环错时
  generated_bank.py       # 仅TRAIN；固定checkpoint、条件、噪声
  model.py                # 独立X/Y编码与双head
  losses.py               # 分证据类型pair ranking
  train.py                # 三组4k更新、恢复状态
  evaluate.py             # 强/弱/生成器外推、捷径探针
  calibrate.py            # 仅RM_cal
  selection_probe.py      # 已缓存16候选排序，无新RL
  report.py
runs/reaction_reward/v1/
  SPLIT_MANIFEST.json
  LABEL_PROTOCOL.json
  GENERATOR_PROVENANCE.json
  SAMPLING_SCHEDULE.json
  RESULTS_RM.md/json
  FINAL_DECISION.json

返回：真实数量、UNKNOWN率、有效偏好数、各类结果与信赖区间、与unimodal探针对比、未见生成器表现、候选选择的真实质量/多样性、不读取DEV的选择依据、是否满足进入阶段B条件。
不要把pairwise presentations称为独立样本；不声称已经有人类奖励准确率。

## 15. 依据与出处

仓库（本轮读取）：
- mam_target/data.py：same-session alternative完整长度线性变形，非真实逐帧配对。
- reaction_flow/condition_encoder.py：三路输入768/25/58以及冻结ConditionEncoder接口。
- experiments/nft_full_test/README.md：本地TEST与VAL内容重复的已有审计。
- generator_reward_nft/train.py / rewards.py：当前NFT采集/更新与手工奖励。

相关原始研究（不是REACT结果保证）：
- REACT2025 challenge, https://arxiv.org/abs/2505.17223
- ImageReward, NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html
- Debiased Contrastive Learning, NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html
- Deep RL from Human Preferences, NeurIPS 2017, https://papers.nips.cc/paper/7017-deep-reinforcement-learning
- Scaling Laws for Reward Model Overoptimization, ICML 2023, https://proceedings.mlr.press/v202/gao23h.html

附件reference_contracts.py及测试只验证数学/数据契约，不包含训练网络、不加载REACT权重、不证明标签正确性。
