# REACT2025：证据锚定的语义事件标注与条件反应规划

## 0. 目标与当前边界

用户目标：同一模型、同一推理策略下，FRC 高于锁定的 MAM 参考，exact FRD 低于同人口的 MAM 参考，同时具有高有效多样性。描述符评分、teacher 标签或 oracle 结果不能代替这三项目标。

本文件是新研发方案，不是已经执行的标注、训练或新性能结果。
本轮只读了当前 GitHub 的 `reaction_flow/condition_encoder.py`、`mam_target/data.py` 和官方数据说明，没有直接检查本地媒体内容。

当前已验证的事实：
- source encoder 显式接收 768-D audio、25-D speaker attributes、58-D 3DMM，没有独立文本/事件输入；这不意味着音频特征完全没有语义。
- TRAIN alternatives 目前先线性拉伸到 source 总长度，再取相同 crop；代码明确标为弱长度对齐，而不是真实逐帧配对。
- 官方 MARS 按同 session 定义适当反应池。新标注不得偷偷重定义官方评价 target 或宣布同 session 标签为负例。

研究假设：补充可追溯的 source 语义/事件时间，以及真实 listener 的反应计划，可能比继续调整噪声尺度更直接地改善适当性与有效多样性。此假设需要 source-only 推理实验验证。

## 1. 首选方向

X -> source-only transcript/events A(X) -> stochastic reaction plan Z -> continuous reaction generator Y。

Z 不是固定 candidate identity：每个候选独立采样一个计划；同一序号没有固定情绪；计划可以相同，不强行去重。第一版继续使用独立先验的 T0，不同时引入 G1 shared-noise、动态正则、assignment、额外 rho 搜索。

外部模型首先作为离线标注教师，提供结构化、有证据的标签，不直接生成连续 25-D 真值，也不当作唯一评价者。

## 2. 三条标注管道必须物理隔离

### A. Source-only 管道：推理可以使用
输入仅允许本任务合法的 speaker 音频、speaker 视频、由它们产生的转录与时间信息。
禁止读取 listener 视频/音频、listener attributes、未来 online 输入、该样本官方 target 池、模型候选分数。

标注内容：
1. word/utterance 时间戳与文本；
2. 可核验语音事件：停顿、笑声、重读、语速变化；
3. 低粒度语义事件：提问、陈述、请求、评价、叙事转折、明确的正/负内容披露；
4. 事件的证据片段、unknown、可见/可听质量。

不要将正面词语等同于真实情绪；不要从脸或声音推断人格、诊断、敏感身份。
批次中的 source ID、session ID 只用于数据管理，不作为教师猜测内容的输入。

### B. Listener observations 管道：只允许 TRAIN
输入：真实 TRAIN listener 的原始 25-D 轨迹、时间轴和可用视频。
数值轨迹用于定位 onset/apex/offset、幅度与占比；视觉模型辅助命名可见动作并识别遮挡/估计异常。
输出观察描述，不写心理事实：“嘴角上扬”优先于“真心开心”，“皱眉可见”不等于“不同意”。
未观察到变化是合法值；没有可见动作不能被标为 missing。
头部动作可作为辅助解释，但当前主输出是 25-D 属性，不宣称头部标签直接监督了并未预测的头部姿态。

### C. Relation 管道：只允许 TRAIN
在真实 paired interaction 中建立 source event 与 listener event 的时间关联。
区分 temporal association 与因果关系；可有多个候选触发源，可为 unknown。
在两段不同 TRAIN speaker 中匹配同义且有证据的事件时，只把 source 事件的内容和顺序交给匹配器；不根据 listener 的 CCC/FRD 倒推匹配。
不要使用 DEV/TEST 的 listener 内容构建任何 source 输入或训练缓存。

## 3. 首批数据与标注质量

先核对本地合法媒体路径、真实 FPS/PTS、音频采样率、音画同步偏移、split 和 recording identity。
不能把 768-D 音频 feature 当 wav 送给 ASR，也不能假设所有样本都是固定 FPS。
如果 source 音频混入 listener 发言，先明确本任务允许的输入范围；禁止通过对方音轨泄漏目标。

建议初始试验：按 TRAIN session 与内容类型分层选 200 个 speaker-event 窗口及对应 paired listener，50 个窗口双人独立复核，分歧裁决。数量为计划值，不是已标注数据量。
包括安静/无明显反应、遮挡、ASR 困难样本，而不只挑表情夸张的窗口。
未给出人工标注授权/接口预算时，先运行本地示例与导出待审任务；不要自动收费 API 或上传受限人脸/音频。

时间检测分工：
- ASR/VAD/forced alignment 提供语言时间锚点；
- 25-D 原始序列的数值检测提供动作候选边界；
- 人工或视觉教师负责语义解释与核验；
- 视觉教师不得凭空给出毫秒级精确 onset。

质量记录：转录错误、事件识别 precision/recall、时间边界误差分布、unknown率、两位复核者一致率、各类别覆盖。
模型自报 confidence 是未经校准的分数，不是概率。接受阈值仅在 TRAIN 人审校准集确定。
不以多轮自洽投票直接替代人审；教师相互可能共享偏差。

## 4. 事件级替代标签：高价值、但单独验证

假设 source i/j 都有相同语义事件，分别发生在 e_i/e_j。
先通过转录/事件序列找对应，而不是简单匹配视频相对进度。
对 j 中 listener 的观测反应，记录相对于其真实 speaker 事件的延迟、幅度和持续时间。
把这一模式用于 i 时，时间锚点来自 e_i；延迟作为弱监督样本或区间，而非精准真值。

第一版优先用 event-relative crops 形成训练样本，不立即把整条 reference 任意重排。
必须保留真实 paired 原时间轴；允许多对多/空匹配，遵守单调顺序，重复句子需上下文消歧。
不确定/未匹配片段仍可作为 session 级弱分布标签，但不进入精确逐时 CCC 监督。
人工复核 20 对跨录制事件，检查“同语义事件”是否真实成立。
训练中优化对齐，不改变官方 Processor、评价 target、长度或全部 K 候选。

## 5. 计划 Z 的最小表达

Z = {global_state, events, no_reaction, uncertainty}。

- global_state：从实际 TRAIN 轨迹计算的低维幅度/占比摘要，例如 AU 活跃占比、VA 分位/均值、expression 时间占比；不是 LLM 编造的情绪概率。
- events：稀疏事件列表，每项包含 source trigger ID（可空）、可见反应类型、relative onset/duration、强度区间与置信状态。
- no_reaction：明确允许无新事件，不能硬凑 4 个或10个不同计划。
- uncertainty：保持多种可接受解释，unknown 不硬转某个情绪标签。

全局摘要可先采用连续数值或少量 TRAIN 固定分箱；不把整段 frame-level 轨迹偷偷复制进计划。
定义统计、平滑、阈值和量化只用 TRAIN；不能用 MAM 输出均值作为标签标准。

## 6. 模型接入：先信息，再结构

### 6.1 最小信息实验
保留 T0 generator，新增独立 trainable SourceSemanticEncoder + 带时间的 tokens。
原 encoder 带有 requires_grad_(False) 和 @torch.no_grad()；新分支必须放在独立可训练路径，不能把新层套在旧 no_grad 中导致训练无效。
先将 source transcript/事件作为额外 cross-attention memory，保留原 speaker features。
采用 NULL/unknown mask、语义模态 dropout；没有标注时显式回退，不用 target 生成替代 source token。

### 6.2 计划条件生成
拟合 p_phi(Z | X, A(X))，离散类型可交叉熵，连续时间/幅度使用能表达不确定性的密度或分布离散化。
真实 paired 计划监督具体输入；同 session 替代计划保持弱标签身份，不能把一个 session 的直方图宣称为每个输入的真实模式概率。
条件 generator 读取 X、A(X)、Z 和局部噪声，输出 Y。

p(Y | X) = integral p_theta(Y | X,A(X),Z) p_phi(Z | X,A(X)) dZ。

训练解码阶段，真实计划 Z* 来自同一个目标 Y；模型学习按照计划生成，而不是复制原 R2 的轨迹。
任务适配阶段必须使用预测/采样计划，走完整 source-only 生成路径。
不能抽一个新计划 Z 然后用与它不兼容的任意目标 Y 做重建；教师计划路径与 source-only 采样路径分开计算各自合法损失。
必要的计划一致性损失使用 TRAIN 固定可计算属性/冻结观察器，不让观察器与生成器一起自由串通。
不得在评估时读取真实 listener 计划。oracle-plan 结果必须独立标为 privileged diagnostic。

## 7. 隔离监督增益的实验

不启动十种新网络。先跑以下最小链条：
E0：同预算原模型续训（或相同容量 NULL semantic 分支）。
E-text：同输入的带时标 transcript 特征，检验是否只是语义特征本身有效。
E-event：同文本基础上加事件结构/时间锚点，检验结构标注的增益。
后续确认：E-plan = E-event + 仅由 source 推断的随机计划。

首次执行优先 E0/E-event；E-text 作为必要的信息来源对照；E-plan 在标注可用性和 source-only 接口通过后实施。
不要把 event retiming 与 planner 一起首次开启；它们需要分开对照。
预算、样本与裁剪调度、父checkpoint、K、solver、原生输出域相同，预登记后执行。
可用 oracle 计划检查 generator 是否能利用标签，但 oracle 成绩不得成为论文主表或超越MAM结论。

机制检查：
- 同 session 内打乱 source 语义，检验是否仅仅记住 session；
- 文本不变但打乱事件时间，检验时间锚点价值；
- 固定 X 与轨迹局部噪声，仅改变合法计划，检验不同反应是否可控；
- 比较实际部署计划与真实计划的差异，避免 target-plan teacher forcing 的性能被误当作 source-only 性能。

主评价：同一 checkpoint/输出策略的 FRC80、exact FRD20（仍是子集）、S-MSE、FRVar、按组及DC/slow/fast分解、目标覆盖与少量盲审视频。
不按 GT 挑候选/噪声/计划。既有开发80仍称开发集，不改为独立测试。

## 8. 预算最值得花在哪里

优先用人审预算解决 ambiguous trigger、遮挡、转录错位和跨录制匹配，不让人逐帧重标所有已有数值。
不要把“LLM写出10种合理反应”当成10条新观测，也不要均匀采样互相冲突的情绪来制造多样性。
若有采集资源，可另立经同意的小型 pilot：同一合法 TRAIN speaker 片段，邀请多位 listener 独立真实响应。那是新增真实一对多数据，不是伪标注；与本文侧标注实验分开记账。

## 9. 权限、复现与交付

MARS 数据有 EULA；本轮未审读签署版本，也未确认外部模型/云API的许可条款。先核对数据授权、外部模型条款及目标赛道要求，再决定本地或云端标注。不得默认私人媒体可上传第三方。
不从短视频标注人格、敏感属性、心理诊断；仅标注可听/可见事件，主观互动功能必须标明推测和依据。

每条标签保存：clip hash、role/split、时间坐标、教师模型与版本、prompt hash、实际可见输入、证据片段、审核状态、generated vs observed、可否用作推理输入。
测试 dataloader：训练标签可读，验证/测试 source 分支只能读source-only标签；修改target缓存不影响公开sample。

建议新目录：
semantic_supervision/{schema,extract,annotate,review,align,models,tests}/
既有模型/结果只读，新阶段独立输出，不自动push。

首次交付：媒体/权限可用性表，200窗口的提议清单，实际可运行的抽取器，已执行数量和人审导出，标签示例与失败/unknown统计，小规模source-only对照配置。
未执行的外部标注/训练不得填为完成；没有真实标注时示例仅用于schema测试。

## 10. 定位参考（不是“首个文本引导”）

- REACT2025 baseline / official README：同session适当集合，原始视频/音频及25-D属性。
- WhisperX (Interspeech 2023), arXiv:2303.00747：VAD与强制对齐的词级时标。
- CustomListener (CVPR 2024), arXiv:2403.00274：自由文本引导与文本标注listener数据。
- VividListener (AAAI 2026), DOI:10.1609/aaai.v40i8.37567：细粒度描述和强度标签。
- OmniResponse (NeurIPS 2025), DOI:10.52202/085713-4196：文本中介与时标化多模态反应。
- ReactMotion (2026 preprint), arXiv:2603.15083：body reaction多候选适当性层级及偏好训练；不等同MARS面部任务。
- VTG-LLM (AAAI 2025), DOI:10.1609/aaai.v39i3.32341：通用视频模型的精细时标仍需专门验证。

正式论文可争取的主张应由新实验支撑：source事件与真实reaction计划连接后，能否在保持任务适当性的同时生成更多种有证据的反应。不要仅以“使用LLM标注”或“增加一层planner”作为创新结论。
