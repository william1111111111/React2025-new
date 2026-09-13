# 语义监督首批交付：候选抽取与本地待审任务

已读取用户任务书和完整材料包，原包逐文件保存在 `semantic_supervision/vendor/react_semantic_supervision`，附件SHA见 `ATTACHMENT_HASHES.json`。这是标注准备阶段，不是已训练的新方法。

## 实际执行

| 工作 | 实际数量/状态 |
|---|---|
| TRAIN文件存在性清单 | 1660条、17 session，所需文件无缺失（沿用预检） |
| 本批录制 | 每session两条，共34条 |
| source/listener视频逐帧PTS读取 | 各34条；校验单调、实际帧数；末帧时长来源显式记录 |
| source候选审查区间 | 937个音频时间窗候选，选200个待审窗口 |
| 200窗口分层 | 低/常规/高音频能量66/70/64；不是已知语义内容类别 |
| 双人独立复核 | 已导出50窗口、250个总答卷槽位；实际人审0 |
| 跨录制匹配待审 | 20任务、17对录制；3任务以另一A窗口再次检查同一对；实际匹配0 |
| listener数值变化候选 | 8473个，全部ambiguous；不是8473个已确认可见动作 |
| 已接受speaker语义事件/paired关联 | 0 / 0 |
| 教师、收费API、GPU、语义训练 | 均未调用/启动 |

200个窗口按speaker音频能量和时间提议，选择不读取listener或候选分数。窗口最长8秒，可能相互重叠，不作为200个独立人物；语义内容、遮挡、ASR困难分层都还需实际审核后补齐。低音频能量不是已验证停顿、无语音或无反应。

source TXT可复用，但没有词/句时标。没有运行ASR、forced alignment或语言模型，没有按文本长度编出时间。speaker语义events保持空，provenance记录明确原因，不把空列表解释为已观察“没有事件”。

listener候选使用已有TRAIN25维轨迹，按真实PTS做200ms平滑，以本批TRAIN数值速度90分位提议变化边界。onset/end/apex引用具体帧和PTS，仅是数值检测证据；不命名心理状态或伪称视觉教师已确认。均值/q10/q90是原轨迹的低维摘要，不是LLM生成的概率或逐帧伪真值。碎片化候选较多，需要人审合并/剔除，不能直接当成稀疏反应计划训练。

## 实际隔离与来源

使用bubblewrap独立进程及mount/PID/network namespace，清空继承环境后只设置必要运行变量。source阶段只读挂载TRAIN speaker WAV、视频、TXT；listener阶段只读挂载TRAIN listener视频/属性；relation阶段只读两侧TRAIN产物。只允许写各自输出目录。不挂载host数据根目录、其他角色媒体、凭据或NVIDIA设备；网络隔离。实际沙箱探针和完整挂载命令保存在各stage日志与 `sandbox_command.json`。这比仅检查JSON中visible_roles更强，但仍不证明原speaker音轨没有串音或角色混入。

实际可见输入哈希、角色、PTS、教师版本(null)、参考prompt哈希和pending审核状态保存在provenance。Prompt哈希仅表明对应材料包提示词，未实际调用教师。音画偏移、发话角色、视觉遮挡、真正no_reaction仍为unknown/未核验，不能用文件名、帧数相同或教师自报confidence代替确认。

`private_recording_map.json`含原ID及session，仅供本地操作者；审核任务使用不透明ID，教师payload不包含session、原描述性文件名、listener或模型分数。没有媒体上传，没有把原始媒体复制到审核包。初次启动遇到本机bubblewrap不支持--clearenv，改为subprocess显式空环境后成功；失败日志保留。首次抽取产物也保留在 `extraction_attempt_001`；最终规范化来源角色的重跑结果在speaker/listener/relation目录，不重复计数。

## 校验和模型入口

- 材料包原7项测试本地通过；新增来源、split、证据哈希、审核/角色/同步门控及target-cache独立性10项测试通过。
- 当前102份sidecar都通过原schema及角色规则（34 speaker、34 listener、34 relation）。这只验证格式/来源声明，不证明事件语义正确。
- 当前34条source记录全部经模型缓存入口返回NULL。只有审核accepted、来源哈希一致、speaker角色与同步核验完成、证据已注册的source记录才可进入source分支。listener/关系不能作为该入口输入；synthetic示例默认拒绝。
- 该入口是缓存访问控制，不是完整新generator；当前模型public sample未改动，语义训练未启动。不能把缓存单元测试夸大为已完成E-event模型端到端验证。

## 交付入口

- `review/proposal_windows.jsonl`：200窗口；`speaker_tasks.jsonl`、`listener_tasks.jsonl`、`paired_tasks.jsonl`分开。
- `review/reviewer_answers.template.jsonl`：250份未填写槽位；`adjudication.template.jsonl`：50份待两人独立完成后裁决的槽位。
- `review/cross_recording_20_tasks.jsonl`：仅speaker的待审匹配任务，match_status=null，不预设同义事件。
- `review/quality_metrics.json`：所有真实准确率、一致率、边界误差为null；unknown100%表示尚未审核，不是模型错误率。
- `semantic_supervision/models/experiment_config.json`：T0父模型、E0/E-event优先和必要E-text对照的配置草案；同容量NULL分支、独立可训练语义分支、固定官方评价。预算尚未预登记，training_launch_ready=false；E-plan与event-retiming均未实施。

后续先做本地角色/同步核验、已有转录强制对齐和这批人审，再冻结TRAIN接受规则、语义分支预算及对照。任务书明确“未给出人工标注授权/接口预算时，先运行本地示例与导出待审任务”；本次已完成该阶段。签署的数据授权、教师模型条款和人审资源仍需确认，未默认启用外部服务。

已有B0/B1继续按原有限预算收尾。没有扩展rho/lambda搜索，没有覆盖历史结果、修改官方target/Processor/FRC/FRD或自动push。
