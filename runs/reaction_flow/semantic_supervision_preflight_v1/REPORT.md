# 语义监督路线：本地数据预检

本次根据用户粘贴的方案开展数据预检。当前附件目录及仓库未找到 `CODEX_SEMANTIC_SUPERVISION.md` 或所述执行材料包；没有假定已读到附件，也没有复用其声称通过的7项测试。

## 已核实

- TRAIN有1660条speaker记录、17个session。所有记录均具备speaker原始WAV、裁脸视频、TXT转录、25维属性、768维音频特征、3DMM，以及同名paired listener属性和视频。完整存在性清单见 `train_inventory.csv`。
- 每session按文件名顺序取两条，合计34条用于媒体元数据抽查。该选择不依据GT质量或模型成绩，不等于200个已标注事件。
- 34条speaker视频均为30 FPS；speaker及listener的视频帧数均与各自25维属性长度匹配。媒体哈希、帧数、音频时长、视频起始时间和特征shape保存在 `pilot_media_metadata.json`。
- 抽查的34个speaker TXT都未发现时间标记。既有TXT是可复用的转录起点，但词/句时间需要单独核验。已有 `whisper_batch_log.csv`，不能仅凭其文件名证明转录准确度或完整模型来源。
- 帧数相等不是声画同步、跨角色同步或事件对应已经成立的证据。后续应以视频PTS/音频时间轴及人工抽查验证；不对真实paired轨迹做全长重对齐。
- `ConditionEncoder.forward`冻结且使用no_grad；新增语义路径必须独立可训练。`TaskData`的同session替代反应确实采用完整轨迹线性插值到source长度，再取crop，只是弱长度对齐。

## 分离的输入边界（尚非已运行的标注器）

1. `speaker_stage_inputs.jsonl`：仅TRAIN speaker音频、视频、TXT及其哈希。不得读取listener观察或关联标签。文本可能包含轮流发言，目录名不能代替说话角色核验；事件发话者身份不确定时保留unknown。
2. `listener_stage_inputs.jsonl`：只提供TRAIN listener属性/视频，显式标记inference_allowed=false。未来仅提议可见变化窗口，不生成伪25维真值，也不把心理解释当事实。
3. `paired_stage_inputs.jsonl`：只记录TRAIN paired ID关系，等候已核验事件和时间轴。跨录制匹配默认unknown，不从同session直接推断对应成立。该文件也禁止推理使用。

这些是隔离的输入清单，尚未实现外部标注执行器或完整访问控制。真实listener计划仅作TRAIN监督，oracle-plan输出必须单独标记，不能混入source-only正式成绩。

## 下一步与缺失材料

请补充原任务书及材料包，以采用其中真实schema、提示词和校验代码；目前不重建一个可能与材料包不兼容的格式。

先对既有speaker转录做小样本内容/说话角色核验、词句强制对齐与音视频时间轴检查，再建立speaker事件及paired事件关联；随后才验证source-only事件输入。200个事件窗口、50个双人复核仍是提案，没有完成任何人工标注。34条抽查录制只是媒体可用性样本，不自动指定为最终200事件人审集合。

签署的数据使用协议与外部模型使用条件未提供；本次只做本地读取，未向第三方上传媒体，未调用外部或付费API，也未下载/运行教师模型。后续教师选择与可用权限需有依据。ASR角色与时间、跨录制关联允许unknown；模型confidence不是校准正确率。

现有B0/B1按已授权2000步预算和固定评测收尾，不继续扩展rho/lambda搜索。没有停止现有训练，也没有修改官方target、Processor、FRC、FRD或已有实验文件。没有新增语义模型训练。

复现：在仓库根目录运行 `.venv/bin/python runs/reaction_flow/semantic_supervision_preflight_v1/audit.py`。本次只有媒体/路径预检，没有语义标签、事件匹配或性能结果。
