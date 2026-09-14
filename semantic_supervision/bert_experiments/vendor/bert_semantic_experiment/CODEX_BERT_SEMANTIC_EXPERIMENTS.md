# REACT：BERT 语义与时间关联的受控实验

版本：2026-09-14。状态：实验提案，尚未实施模型训练。
本文依据 GitHub `william1111111111/React2025-new` 的 `38615960d2235f177c5cec1dc36b14d5de6929e6` 制定。执行前读取真实 HEAD 和本地进度；已有更晚结果时复用真实结果，不覆盖历史。

## 0. 目标与本轮边界

用户最终目标：同一 checkpoint、同一种 source-only 推理策略，同时具有高于 MAM 的 FRC、低于 MAM 的 FRD，以及高有效候选多样性。
本轮只回答两个更具体的问题：

1. 预训练上下文文本表示替代独立字节平均，是否提供任务收益？
2. 在相同 BERT 内容表示上，显式 frame-event 相对时间偏置是否提供额外收益？

不预先宣布 BERT 必然解决多样性，也不把旧 E-event 点估计解释成语义方向无效。文本分支本身对同一输入的 K 个候选相同；它不是额外的随机模式来源。

不新增 planner、共享噪声、OT/assignment、动态损失、diversity reward 或梯度比例控制。不扩教师调用、不改自动标注门槛、不要求恢复人工标注、不把 unknown 当 listener 或 no-event。默认 seed123，不自动做多训练 seed。

本轮进行本地独立开发分支 `agent/bert-semantic-controls`；不自动 push，不覆盖旧结果。不得将本任务书或代码中的样例、计划当成已完成实验。

## 1. 已核对的起点

- `semantic_supervision/models/semantic_flow.py` 使用 `Embedding(257,d).mean(0)` 编码 UTF-8 字节；文本内部没有上下文编码或位置建模。
- `semantic_supervision/experiments/controlled.py` 使用旧 T0 的 FM + `lambda_ref * task_parts`，500 步受控实验，48 条 TRAIN source cohort。
- `runs/reaction_flow/semantic_controlled_v1/PROTOCOL.json` 是已执行实验协议；旧 `models/experiment_config.json` 的未启动状态不能覆盖它。
- 同目录 `DEV_READY.json` 记录 80 条 source、48 条有非 NULL 语义、172 个弱事件。这是 source 级存在性，不是所有帧都有事件。
- `event_typer.json` 记录同一 TRAIN 拟合分类器，但 TRAIN 与 DEV 的 proposal 切分不同。记录这个边界，本轮不同时修改 proposal 规则。
- 不完整 FRD 只标 pending；不将中间配对均值作为最终 exact FRD，不重算已经完整且身份匹配的历史 FRD。

共同父模型：
`runs/reaction_flow/task_dynamics_v1/T0-task/attempt_000/checkpoints/step_014000.pt`
SHA256：`e6199183cdc3526e1fc1390c89057e40514af8d2f2101d69a5e1dabb7f43f4d6`。

现有 parent FlowConfig 为 d_model=256、heads=8、T=750、B=4。以实际 checkpoint 配置核对，差异显式报出。

## 2. 必做四组实验

| arm | 内容编码 | 起止时刻作为 token 特征 | 显式 frame-event 时间偏置 | event type |
|---|---|---|---|---|
| P0-null | 无实际语义，校正为零 | 不向预测注入 | 无 | 无 |
| P1-byte | 可学习独立 UTF-8 字节向量的平均 | 有 | 无 | 无 |
| P2-bert | 冻结 BERT 上下文表示后池化 | 有 | 无 | 无 |
| P3-bert-time | 与 P2 完全相同 | 有 | 有 | 无 |

主要比较：P2-P1 为上下文预训练编码替代字节统计的实用收益；P2-P0 为整体显式文本条件的实用收益；P3-P2 为显式相对时间机制收益。

P2-P1 同时改变表示结构与预训练资源，不得称严格隔离了“仅预训练权重”的效果。若后续需要该主张，另加一个 frozen random-BERT、同 tokenizer/配置/融合器的受控 arm；不要默认运行。

所有组使用同一新融合实现、NULL 规则、输出零初始化、优化器日程与样本日程。旧 E0/E-text/E-event 只作为历史参照，不能将旧 E0 与本轮新融合器直接称作同架构控制。

P1 的字节嵌入输出 768 维，以复用共同的 768->256 投影；因此它不是历史 d=256 byte 模型的逐参数复刻。必须报告各组总参数、冻结参数、实际优化参数和推理成本，不能宣称所有组参数/算力完全一致。P0 实例化共同融合模块但语义路径不激活，这是合法的缺失输入基准，不是活跃参数数目的严格匹配。

## 3. BERT 编码契约

### 3.1 模型

- `google-bert/bert-base-uncased`，英文，12 层、768 隐藏维、12 注意力头。
- 使用本地 `AutoTokenizer` + `AutoModel`/`BertModel`，不是 MLM logits。
- 下载前解析并记录真实模型 revision；不要使用未记录的可变 main，也不要虚构 SHA。
- 记录 tokenizer、config、权重摘要和 transformers/torch 版本；使用可信 safetensors，`trust_remote_code=False`。
- 全部 BERT 参数冻结，固定 eval 模式；生成器 `.train()` 不得重新打开 BERT dropout。
- 可在单独进程预提取缓存，之后训练时不再加载 BERT。若容器/网络无法取得权重，不用随机初始化冒充 pretrained。

### 3.2 输入

只编码经过现有合法 source-only auto_weak 门槛的 `WeakEvent.text` 原始英文证据。第一轮不输入 LLM 中文 description、不加事件类别、不加 session/clip/人物 ID、文件名或 GT 信息，不把未知角色全文作为上下文。

每个事件单独编码，避免加入新的邻句上下文成为混杂因素。离线设置允许该合法事件的完整文本，但不能据此宣称 online/causal。

动态 padding。默认每个窗口 128 tokens（含特殊 token）；超过 126 个内容 WordPiece 时，按文本顺序划分非重叠内容窗口，每窗加特殊 token，最后按内容 token 数加权池化成一个事件向量。记录窗口数/长文本比例；不静默截断，不按字符分配时间，不增加或删除 event 槽位。

### 3.3 池化

使用最后一层 `last_hidden_state`；对 attention_mask=1 且非 CLS/SEP/PAD 的内容 token 做均值：

`e = sum(hidden * content_mask) / sum(content_mask)`。

空白/零内容 token 返回缺失，而不是除零。BERT 是“先上下文建模，再池化”，不同于独立 byte 向量直接平均。这个池化是固定实验选择，不声称它是所有任务上最好的句向量方法。

不使用 `pooler_output` 作为默认事件表示；它包含为 NSP 预训练的变换，未针对本任务校准。第一轮不同时搜索 CLS、最后四层、SBERT 等。

缓存保存 FP32 事件向量、内容哈希、model/tokenizer revision、池化/窗口规则、表示版本；标注权限和时间证据继续保留在独立 source cache。相同文本可共享纯内容向量，但不同记录的角色与时间元数据绝不共用。

对冻结编码器，TRAIN/DEV 的独立合法文本特征可分别无梯度计算；任何拟合、尺度学习、分类器更新只用 TRAIN。不要用 DEV 进行 BERT 领域适配。

## 4. 共同融合层：既提供新信息，又不改变起点函数

复用 T0 已有 source encoder，增加独立可训练路径，位于其 no_grad 外：

`English text -> frontend -> e_j[768] -> LayerNorm(768) -> Linear(768,256)`。

P1 前端是独立 byte embedding 的平均；P2/P3 前端是冻结 BERT 事件向量。

对有测得时间的事件，使用同一归一化时间特征：
`[start_rel / crop_duration, end_rel / crop_duration, 1]`，通过 Linear(3,256) 加到内容表示。
对时间未知但角色有支持的内容，使用 `[0,0,0]`；不得编造时间。

在 h[B,T,256] 上做一次 semantic cross-attention，然后：

`h_prime = h + sample_has_semantics * frame_valid * output_projection(attention_result)`。

只将最后 output_projection 的 weight 和 bias 置零，其他投影/attention 常规初始化。不要同时设置一个乘法 gate=0，避免切断双方初始梯度。

验收：在 step0，P0/P1/P2/P3 同输入同噪声的预测与原 T0 在数值容差内一致。第1次 backward 上游语义参数梯度可能为0；只需最终输出投影获得有限非零梯度，并在后续2-3次更新中看到上游参数进入梯度链。不要写“所有参数第1步必须非零”的错误测试。

NULL token 用于避免空 key 的 attention NaN；没有任何保留事件（或被 semantic dropout 整体移除）时显式将语义 residual 置零。任意训练后该组自己的 base 路径应当是 NULL 的精确回退；不要求不同 arm 更新后的 base 彼此相同。

同一 source 的 K 个候选共享同一 h_prime，没有 candidate identity、sample-axis attention 或按目标选候选。

第一轮所有组每样本 semantic dropout=0.1，使用独立、预生成的 Bernoulli 掩码；即使 P0 不使用，也消费/记录同一掩码。FM 和 task rollout 共享本步同一条件结果。

## 5. P3 的显式时间偏置

只改变 attention logits，不换文本、不新增事件、不改 time_projection。
设 q_i 是由当前 crop 的实际视频 PTS 得到的帧时间，事件区间为 [s_j,e_j]，都在同一 crop-relative 秒坐标。

`d_ij = max(s_j - q_i, 0, q_i - (e_j + 2.0))`
`b_ij = clip(- d_ij**2 / (2 * 1.0**2), -8.0, 0.0)`。

P2 使用 b=0；P3 将 b 加到 scaled dot-product attention logits。2秒后向宽限和1秒尺度是冻结的工程先验，不是已知人类反应延迟或生理阈值，不从 DEV 选值。

这是软偏置，不是只准事件发生时响应；前后仍可能响应。NULL token 偏置为0；无可靠时间的合法内容 token 偏置为0并保留 known-time mask；padding token 单独严格屏蔽。

使用视频真实 PTS 和已记录的 audio->video offset，不使用固定30/25fps，也不把 Flow 的生成进度 tau 当视频时间。

主四组使用相同的已裁剪事件列表，保持当前 crop_events 的选择规则。不要只有 P3 额外引入上一个 crop 的事件。跨块边界延迟窗口不完整是该首轮实现边界；若后续增加2秒历史事件，所有相应控制组一起更新且版本分开。

可记录 attention 对NULL、timed、content-only token 的质量分配，但 attention 大小本身不证明语义有效。

## 6. 数据与日程

首轮固定 `semantic_controlled_v1/cohort.json` 的48条合法TRAIN source、对应缓存与当前DEV80缓存；执行前验证实际计数及哈希。

不按 DEV 分数扩大/缩小 cohort，不为了通过率放宽角色、CTC或同步门槛。不将48条称为完整1660条训练集。

四组从同一个原 T0-14000 出发，不从旧 E_text/E_event 的500步模型开始；共同 fusion 初值逐项一致，BERT加载/前端初始化不得改变 base 参数或 source/noise RNG。

统一新增1000个实际优化步；检查点0/100/500/1000。1000是固定的起始实验预算，不是收敛保证，也不是根据某组500步的分数单独加训。

沿当前合法48-source抽样规则生成足量1000条 schedule；可复用原前500条并验证，再确定性延长。不得用zip静默截短。每一步的source、crop occurrence、paired/alternative targets、target slot、FM noise、tau、K4 rollout noise、semantic dropout均匹配。

- TRAIN B=4、T=750、K=4。
- 目标保持 T0 FM + 原 lambda_ref * task_parts，不添加动态/分散度/事件分类损失。
- source encoder 继续冻结；Flow velocity 与语义融合层可训练；BERT冻结。
- 继承共同T0的velocity AdamW状态，velocity lr=2e-5。
- 语义新增参数使用新的AdamW状态，lr=1e-4，前100步线性warmup；四组相同。
- weight_decay=.01；相同global gradient clipping norm=1.0。
- 所有超参数是本轮预定方案，不声称最优。不同于历史500步的部分已由四组共同重跑控制。
- 保存完整optimizer/RNG/phase_step/schedule cursor和来源hash，短程断点回归后再正式跑。

记录实际有效帧数、不同source数、不同事件数、重复访问、每crop事件数量、timed/content-only比例、dropout前后非NULL数量、语义更新数及资源开销。

48条反复裁剪与4000次source访问不是新增4000个独立样本。该阶段只做开发比较；扩大高覆盖TRAIN是另一个后续数据阶段，不能与编码器升级混在第一次对照中。

## 7. 评价和干预诊断

### 7.1 主表

保持原DEV80完整序列、K10、固定同一noise bank、FP64 Euler16->FP32官方评价、原生连续AU、目标池/Processor不变。

500步记录FRC/S-MSE/FRVar学习曲线；1000步四组都执行FRC80、exact FRD20、S-MSE80、FRVar80、目标侧覆盖及DC/slow/fast。每行都来自同一checkpoint和策略；FRD20不是完整80-source FRD。

已有E系列未完成FRD允许复用完成的新本地结果，但不混入新P系列主表。等待/队列或中间配对不能算完成。

### 7.2 覆盖分层

在看模型结果之前用固定source缓存建立三组：全80；有非NULL弱语义的source（上次为48）；其余NULL source（上次为32）。再单列确有timed事件可进入crop的覆盖。

各arm都在同一分层上统计；保留全80为主，非NULL子集只解释有效暴露。各arm的base网络也更新了，所以NULL子集不一定与P0有相同输出。不能把全表简单视为60%语义收益加40%恒定基线。

报告逐source成对差值和session等权摘要。可用session cluster bootstrap报告固定checkpoint上的开发不确定性；不能将帧/候选/重复noise当独立人群，单seed不提供训练随机性稳健结论。

### 7.3 两项有限扰动检查

对1000步的P2/P3用固定同噪声检查，不重新训练，不改变原生主表：

A. 同session跨录制文本错配：保留接收者的事件个数、时间、角色门槛及缺失mask，仅替换为其他合法source的文本/缓存向量；使用预生成固定配对，不看listener或结果。禁止同一个文本字符串的无效置换。没有可用不同文本的组报告不可测，不编造不同事件。主要报告FRC/paired CCC/预测变化，正确输入是否更合适；并非“输出变化”就证明利用有效。

B. 时间错配：保留文本token顺序不变，将不同事件的interval在record/crop内置换；不能把(text,time)整体一起换序，因为attention对key/value共同置换本身可以不变。少于2个不同timed事件的crop不计入该诊断，不当作零影响。固定同一可测source集合比较正确时间与错误时间。无需额外跑所有扰动的exactFRD。

这些扰动是机制线索，不是独立测试，也不是提高分数的新推理技巧。

### 7.4 验收解释

- P2对P1/P0出现更好的任务质量-多样性组合，且正确文本优于固定错配：支持继续投入预训练文本表示。
- P3对P2进一步改善、并对正确时间表现出可检验收益：支持显式时间机制。
- 只看到FRC改善但多样性不增加：将其称为条件质量进展，不宣称已解决高多样性。
- 不通过则先看文本是否实际非NULL、特征是否有区分、时间是否匹配、是否过度适配48条；不直接上planner或把所有失败归因于数据。

开发筛选可预登记：候选FRC不低于同终点P0，FRD20不高于P0的1.02倍，S-MSE不低于P0的0.95倍；这些只是一组工程容差，不是统计等效或论文优势声明。最终选择还应报告所有点和不确定性，不通过时不事后修改门槛。

用户最终MAM联合目标始终保留：当前匹配开发参考 FRC=.810962319、FRD20=172.576423473、S-MSE=.157203704。先验成本/原生AU策略不一致要如实记录，不能把小于MAM的多样性称为全部达标。当前数值不能外推隐藏测试。

## 8. 只有主四组完成后才考虑的后续

1. 若需隔离预训练权重：同架构同tokenizer的冻结随机BERT对照，其他不变。
2. 若冻结BERT已有信息收益、TRAIN支持量足够：从同一P2或P3父模型，比较继续冻结与只解冻BERT顶端2层，BERT lr=2e-6，其余共同；不要只给解冻组多训练。解冻后禁止复用静态BERT缓存；重新在线前向或分层缓存必须与梯度一致。
3. 若语义已提高质量但多样性仍欠目标，再研究TRAIN真实反应计划。BERT是确定性条件信息，不是“十种答案”的直接生成机制。
4. event type是后置消融；现有TF-IDF伪类型不能直接称强人工标签。不先扫RoBERTa/SBERT/大语言模型。

## 9. 必要测试，不扩成无尽审计

- BERT模型revision/feature shape=768、冻结且eval、缓存重放一致、PAD和特殊token不参与pooling、空文本NULL。
- 字节平均对同一字符多重集合的词序置换相同；BERT不在代码结构上强制相同。句向量差异不是语义准确率，不设臆造margin。
- 事件本身的文本顺序改变，与事件列表(text,time)共同换序的两种测试分开。
- 已知PTS/crop变换和时间偏置的平移不变性；unknown timing不用硬定位。
- 仅zero-init最终projection；第1步输出projection和后续上游梯度检查，不要求首次全模块非零。
- NULL residual=0、padding masked、K-prefix和同noise可复现。
- 每arm所有数据/noise/dropout记录一致，实际步数=请求步数=日志行数；可恢复。
- source-only数据与listener训练目标分离，DEV只输入/评价无训练；不让speaker派生文本携带未审核的其他角色内容。

## 10. 执行交付

建议独立目录 `semantic_supervision/bert_experiments/`，不要覆盖旧SemanticFlow；复用已通过的数据/目标/评价组件。

交付：冻结配置与hash；特征缓存摘要；上述测试；四组真实学习曲线；同终点完整指标；缺失分层结果；两项有限扰动；可训练/冻结参数和成本；一个下一步研究决定。

禁止只交计划和更多审计目录；但也不能伪造没有执行的训练或完整FRD。数据/权重不可用时完成可执行代码及可运行测试，明确未运行部分。

## 来源（用于实现核对，不是实验成绩）

- Repo: `william1111111111/React2025-new@38615960d2235f177c5cec1dc36b14d5de6929e6`
- `semantic_supervision/models/semantic_flow.py`
- `semantic_supervision/models/auto_weak.py`
- `semantic_supervision/experiments/controlled.py`
- `runs/reaction_flow/semantic_controlled_v1/PROTOCOL.json`
- `runs/reaction_flow/semantic_controlled_v1/DEV_READY.json`
- `runs/reaction_flow/semantic_controlled_v1/event_typer.json`
- Devlin et al. BERT (NAACL 2019): https://aclanthology.org/N19-1423/
- Model/config: https://huggingface.co/google-bert/bert-base-uncased
- API: https://huggingface.co/docs/transformers/model_doc/bert
- Outputs: https://huggingface.co/docs/transformers/main_classes/output
