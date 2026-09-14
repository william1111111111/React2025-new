# 自动 Who/When → 独立 auto_weak 输入

起点HEAD为2a5c93e2，工作区最初干净。旧250事件、旧42名单和gold source_cache契约均未覆盖；本轮未push、未收费API、未人工复核、未启动长训练。

## 已执行

1. 只读取旧words/scores/result重算250事件支持度，未为覆盖率漏洞重跑教师。新增expected/overlap/actual/control/common/usable帧数、覆盖率、时间域交叠、时长、连续非发声区间和逐规则拒绝原因。按真实times_s二分搜索，不用裁短数组伪装尾部完整。
2. 固定工程门槛：共同与可用帧覆盖≥0.9、时间域重叠≥0.9、至少5帧、至少0.2秒；保留原时间/角色阈值。排除循环拼接±0.12秒；记录3秒mod音频长度后的有效位移，近零位移不能当有力对照。连续≥0.24秒非发声区间按可能跨话轮拒绝，不将unknown当listener。
3. 原42联合候选保留32、排除10、新增0。排除中7例有明显非发声连续段，2例拼接边界后覆盖不足，1例超短句。事件名单与多标签分层原因见support_report.json，原92 timing-uncertain和116 timing-only可逐项追溯。对7个跨话轮父事件导出10个原词范围拆分提议；不继承父事件类别，尚不直接加入训练缓存。
4. 合法speaker媒体的局部滞后网格[-0.4,-0.2,0,0.2,0.4]秒，非循环移位并屏蔽两端；峰值必须在网格内部、至少25帧且与第二名差≥0.1。偏移仍是estimated/unresolved，sync_verified始终false。保存WAV采样原点、实际PTS原点、offset来源及有效范围；±0.2秒区间是网格分辨率提示，不是置信区间。
5. 独立auto_weak缓存与加载器：英文原文/类别/时间证据、源媒体及教师/策略/版本哈希、硬门槛与拒绝原因。读入时校验外部提供的cache/media/policy身份；拒绝listener来源、错误split、session等额外字段；缺失返回NULL。NULL不是无事件标签。模型只收typed事件，不收文件名、cache路径或listener输入。
6. T0外独立可训练分支：UTF-8字节均值嵌入＋事件类别＋独立时间投影＋attention，位于冻结ConditionEncoder的no_grad范围之外。它是最小接口基线，并非预训练语言模型。time转换采用音频秒+显式offset→真实帧PTS→crop交集；不按字符摊时间。时间域不可靠时仅保留合法角色过滤后的无时标内容，不做精确帧注入。
7. 通过边界/梯度检查后，冻结教师/门槛/探针，对17个session各取另外2条既有TRAIN候选：34条、300事件完成同一链并输出弱缓存。没有重调门槛或调用LLM。FROZEN.json记录当时版本；唯一后续接口修正是缺失cache文件返回NULL，标注/模型代码未改，版本差异见VERSIONS.json。

## 可用性与覆盖

| 阶段 | 录制 | 事件 | 可加载auto_weak | 可在有效范围按估计偏移映射 | 仅内容 |
|---|---:|---:|---:|---:|---:|
| pilot | 33 | 250 | 32 | 22 | 10 |
| expansion | 34 | 300 | 163 | 145 | 18 |

共处理67条录制/550事件，195个非NULL弱事件；不是1660条全量Who/When。coverage不足session保持缺失，分session分布见SUMMARY.json。CTC与同模型greedy agreement是相关自洽证据；ASD是自动音画关联，非身份真值，均不解释为校准概率。

## 真实 mini-batch 证据

T0 SHA：e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6。实际TRAIN B=2、64帧crop，英文引用、相对秒、交集帧索引及PTS哈希详见mini_batch_result.json。

- 原32条实际全部通过新loader，真实mini-batch含非NULL且有时间的事件。
- semantic各参数梯度非零；冻结ConditionEncoder无梯度。新增projection与attention实际更新。
- 内容/类别扰动最大输出差0.247429，纯时间扰动差0.0110724。
- 同输入/同噪声重复误差0.0；K-prefix最大误差1.75834e-06（FP32容差1e-5）。
- 仅semantic分支3次AdamW更新，T0本体冻结；原FM损失只在训练驱动层接收paired listener，未进入semantic缓存。诊断损失[0.3945448398590088, 0.3919770419597626, 0.3901004195213318]；不据此声称任务指标提升。
- 26项测试通过：包括100预期帧仅1有效、全NaN、视频部分覆盖、超短、越界、非零原点、crop/尾部、角色unknown/缺标NULL、时间组件隔离、listener文件变更不影响semantic接口。

## 未执行与冻结边界

未做正式E0/E-text/E-event训练或FRC80/exact FRD20/S-MSE/FRVar评价，未训练planner/搜索rho或loss。auto_weak_experiment.json预登记同父模型、同有效数据、同预算/噪声、同容量分支三组，预算尚未设置。没有因小样本短程结果判定路线成败。

真实同步/身份仍有模型不确定性。非零视频原点的纯覆盖/映射逻辑有测试，教师包装器遇到此情况仍保守拒绝，未伪造同步。现有真实选样原点为零。未来DEV须只跑冻结合法source链，不用gold或listener替代；现有transcript辅助设置不是音频-only，部署缺转录需要另行实现ASR/蒸馏。

## 产物入口

support_report.json / support_events.json；cache/index.json；expansion/weak/cache/index.json；lag/；turn_split_proposals.json；mini_batch_result.json；SUMMARY.json；VERSIONS.json。
