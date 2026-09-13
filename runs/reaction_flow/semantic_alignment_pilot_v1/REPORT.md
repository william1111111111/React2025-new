# 自动 Who / When 标注试点

依据用户附评审，只推进既有候选的角色与时间标注；用户明确省略人工。未扩写语义、未训练E-event或planner。

## 结果

33条TRAIN录制、17个session、250个原有事件全部完成本地自动处理。

- 158个事件通过自动时间证据门槛；92个时间不确定。
- 42个同时具有时间及主动说话人模型支持，标为source_speaker_candidate；208个角色unknown。
- 人工复核0，训练可用标签0。42个是自动弱候选，不是经核实的身份或同步真值。
- 没有上传媒体，没有付费API调用；只使用一张GPU。

## 时间与角色依据

Wav2Vec2 ASR BASE 960H对真实16kHz重采样音频做CTC强制对齐，字符路径合并为词，原文引用映射为事件区间。20ms特征步长；时间不是毫秒级精度承诺。模型按20秒核心加1秒上下文分块推理，保留全局采样格点。字母/撇号规范化；含数字引用不通过时间门槛。与同模型贪心ASR一致性仅为相关的自洽检查，不是独立ASR核验。

TalkNet-ASD pretrained TalkSet使用speaker视频实际PTS采样到25Hz，按上游224像素缩放与112中心裁剪规范、2/4/6秒尺度计算原始主动说话分数。音频循环错移3秒作为诊断对照，不修改真实时间轴；循环边界、其他持续说话片段和模型域偏移可能影响该对照。分数与CTC后验均未经本数据校准，不解释为正确概率。

预先记录POLICY.json：CTC词分数平均至少0.5、ASR字符一致性至少0.6、单词时长不超过2秒；角色还要求正logit帧比例至少0.8、相对错移对照平均logit差至少0.5。阈值是启发式，没有用指标或listener挑选阈值。代码/权重哈希见job/job.json。

## 权限与限制

bwrap独立网络命名空间，只挂载TRAIN speaker音频/人脸视频/转录、只读本地教师及选定候选；没有挂载listener目录、主数据根目录或API密钥。仅挂载单GPU设备。启动时存在一次GPU可见性失败，保留execution.log；改用GPU UUID及只读动态链接缓存后成功。

实际音视频偏移尚未被独立验证：使用各容器原点，明确标记container_origin_assumed_not_verified。ASD角色是画面人脸发声关联的模型提议，不是说话人身份真值。没有把人工省略改写成review_status=accepted，也没有放宽原有正式source_cache门槛。

## 产物

- events_all.jsonl：250个事件及时间/角色不确定性。
- events_auto_weak.jsonl：42个自动弱候选。
- source/<id>/words.json：逐词采样时标、分数和原文字符范围。
- source/<id>/active_speaker_scores.json：原时间及错移对照分数。
- source/<id>/result.json：输入哈希、来源权限、教师哈希和限制。
- SUMMARY.json：按session统计。

3项新增测试覆盖重复字符CTC路径、重复引用定位、改写引用拒绝；另有10项既有来源边界测试。测试验证实现边界，不验证真实语义或身份正确率。
