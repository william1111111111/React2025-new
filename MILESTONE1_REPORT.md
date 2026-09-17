# Milestone 1 — 前置交付与最小实现

2026-09-17。**STEP 1–5 已完成准备，STEP 6–7 已实现最小模块；完整 MILESTONE 1 尚未完成。** 先完成并汇报前置核查，收到用户“继续”后完成最小实现。仅运行12步CPU codec集成检查；没有正式大规模训练、API 请求、媒体上传、旧生成器更新、Q0/Q1 恢复或 git push。

## 已交付

| 步骤 | 实际产物 | 状态 |
|---|---|---|
| 1 方法核查 | `METHOD_RECON.md`、`runs/reaction_program/milestone1_v1/RECON_EVIDENCE.json`、逐维 `CHANNELS.json` | 完成当前可核实事实与未知项 |
| 2 统一评价 | `runs/reaction_program/milestone1_v1/evaluation/manifest.json`、`LOCK.json`、fail-closed loader | 1142双方向 / K10 / R10 成员、次序、输入和预期处理后目标哈希已冻结 |
| 3 annotation schema | `annotations/reaction_program_v1/schema/`、真实 pending sample、分折/权限校验 | 四类格式及 raw response receipt 已提供 |
| 4 TRAIN 候选 | `annotations/reaction_program_v1/train/*.jsonl` | 600 event candidates，来自600条录制 |
| 5 输入与 prompt | `annotations/reaction_program_v1/prompts/`、`train/requests/`、`PROVENANCE_ENVELOPES.jsonl` | 1494请求已准备但未发送；另有 paired association prompt |

代码：`reaction_program/preparation/`。数据目录 train / val / test 物理隔离；后两者目前为空，不从验证/测试 listener 拟合 ontology。

## 评价核查的实际发现

1. 现行旧仓库 `dataset/react_2025.py` 已是475行版本，test的 `k_select=9`；本仓库 DEV80、近期 full VAL / local TEST 清单均为 **paired+9**。旧 >1100行 loader 的审计文档曾写 paired+10，不能沿用为当前事实或混合分数。
2. 新主清单复用已有1142双方向确切reference成员，映射至内容一致的 VAL，**3426个唯一 source/target 特征文件已逐文件验证SHA256**。不是重新抽目标以改变成绩。原DEV80继续单列，不能与全量主清单拼表。
3. manifest SHA256：`61dd1d3fc599a6e9ef324632839d4c8df454bd6d0bc26c5b8d0f97d80df1065d`。
4. **1142份 Processor target 数组此前被清理，当前全缺失。** 预期哈希、原始目标、官方 Processor源码/config/checkpoint与seed策略已固定。评价入口会拒绝缺失缓存，不临时插值或重抽reference。下一次实际EXP-A前，须用旧实现恢复并对照哈希；如不可完全复现，需另立共享缓存版本并所有模型共同重评。此次没有冒称重算EXP-A。
5. 本地TEST内容等于VAL，是复用开发数据，不是隐藏确认集。
6. P2近期固定checkpoint SHA256为 `780d761e9efd7bfbc5a32dc12009928cc1b686e6df90d92c3b773b886a1d9986`。S0有更高DEV80 FRC、MAM有更高S-MSE，不能把P2叫作三指标都最强。

### 历史 P2 完整评价（不是本轮新成绩）

| 人口/输出策略 | FRC | 完整 exact FRD | S-MSE | temporal S-MSE | FRVar |
|---|---:|---:|---:|---:|---:|
| 本地TEST双方向1142、K10、全NULL语义 | 0.786617945 | 127.926324864 | 0.066014679 | 0.056890124 | 0.053226778 |

该行来自旧 `COMPLETE_RESULTS.json`，原始scope完整保留。S0 step8000的DEV80/FRD20为1.221969005 /133.312706288 /S-MSE0.043695860，和上述全量不能直接比较。

## 首批候选覆盖与证据强度

- 600条来自TRAIN RM_fit，600个不同recording、17个session、17个保守录制日期连通组。
- 每个session35或36条；seed123/hash固定排序，一条录制选一个event。原始有逐字证据的可用recording池1088条。
- 已有自动角色支持：48条；可映射source帧区间：34条；角色仍unknown：552条。
- 时间映射保留`estimated`和实际PTS来源，不宣称独立同步验证；anchor_frame未观察则null。
- Listener真实媒体待观察任务600条；**已完成动作观测0**。actions=null不是maintain，不能参与executor监督。
- 同session跨录制equivalence候选294对；**确认匹配0，observed_support=0**。候选配对本身不是语义等价。
- 600条原始listener数值仅做数据契约审计（不用于选择speaker事件）：AU均为0/1；expression概率和最大误差2.38e-7；帧长153–8580。
- 15 AU及8 expression的确切类别顺序未定位；52 expression coefficients也不能直接解释为52个动作。逐维未知项如实保存。

因此这不是“600条高质量动作标签已完成”。得到稳定可解释 program 的Q1，以及有语义等价事件但不同实际program的Q3，都仍需真实视觉观察、等价判断与核验。

## Annotation / ontology 设计

- Speaker输入只含source转录/媒体/既有时间角色证据，不能看到listener；原始引文、字符范围、teacher/prompt/input/hash留档。
- Listener先独立做open-vocabulary可见动作观察，保留叠加动作及原始时间轴；随后单独关联paired speaker event。
- Equivalence annotator不看到session/文件名/listener；session只负责产生candidate pairs。
- Private resolver储存执行路径与身份映射，不能拼入teacher或planner输入；请求只传opaque asset IDs。
- 每次未来实际调用须留raw response、真实返回模型/version、prompt/input hashes、实际访问模态。当前已有DeepSeek文本提议不冒充视频动作标注；未提供revision明确为unknown。
- Ontology v0是开放格式及证据归并规则，不是捏造的数据驱动词表。待获得描述后再embedding/聚类/合并，冻结后才能用VAL评价。
- strict/relaxed与human gold分开。最新任务书要求后续分层人工抽检，已写明分层方法；未伪造人工通过，也没有在本准备阶段要求新增人工结果。

## 验证

- 7个必要单元测试通过：固定R10、缺失缓存拒绝、K10/长度/finite、标识符泄漏、pending observation不能变observed support、同折约束、缺标不等于正支持。
- 2094条四类JSONL记录通过schema和真实数据校验；1494请求通过角色隔离检查；600条引用逐字匹配TRAIN文本。
- 原官方指标使用12帧真实TRAIN数据做最小契约检查，结果另存`OFFICIAL_METRIC_CONTRACT.json`。只验证求和/距离/多样性定义，**不是codec或新模型结果**。

## Milestone问题的诚实答案

A. GT→codec→decode的FRC/FRD损失：未执行，没有数值。
B. GT program+GT timing是否能高质量恢复：未执行，没有已完成program标签。
C. 不同program是否产生显著不同25D：未执行。
D. 是否来自semantic program而非幅度/噪声：未执行；后续必须固定style/noise并做program shuffle/幅度匹配对照。
E. 是否保持真实范围和temporal统计：目前只审计了真实数据，未审计尚不存在的生成结果。

## 下一阶段边界

最小codec/executor已实现并通过接口测试；下一步先解决codec短程检查的低code使用率/量化损失，再做有预算的保真验证。Codec可基于真实TRAIN轨迹独立检验；没有可用动作观测与时间证据时，不用伪造program启动正式EXP-C。正式全量评价前恢复固定Processor缓存。Planner、clock预测、allocator、完整训练均等codec/program可行性证据；不添加RL、GMM或raw donor transport。

## 复现与版本

- 当前git HEAD：`f47c363db`（完整SHA见RECON_EVIDENCE.json）。本轮新代码未提交；实际源码哈希见ARTIFACTS.json。之前vector-selection未提交文件仍保留。
- `.venv/bin/python -m reaction_program.preparation.build`：独占创建数据/manifest，已有输出会拒绝覆盖；重跑用新版本目录。
- `.venv/bin/python -m reaction_program.preparation.schema`：生成schema。
- `.venv/bin/python -m reaction_program.preparation.validate`：真实数据校验并保存结果。
- `.venv/bin/python -m unittest reaction_program.preparation.test_contracts -v`：7项契约测试。
- `.venv/bin/python -m reaction_program.preparation.finalize`：冻结prompt/input/源码证据与baseline。
- 本轮仅CPU，正式训练步数0；codec集成检查12个优化步，executor优化步0/反传检查1次，API请求0。交付包含文本/索引/PTS，不复制原始媒体或权重。

## STEP 6–7 最小实现与真实检查

代码：`reaction_program/{motion_codec,rvq,reaction_program,reaction_clock,motion_generator}.py`。Codec为stride4 / codebook512 / RVQ2层 / latent128，按AU/VA/expression分块损失并加CCC、velocity与量化项；不直接拼接83D。Executor为显式多action+GT timing的masked-token Transformer，支持重叠动作；没有planner和预测clock。

新增6项测试通过，连同准备期共13项。实测覆盖尾部mask、有效梯度、token编码解码一致性、program/clock扰动改变输出、显式随机流可重复与K-prefix。扰动检查在未训练网络上，只证明计算路径，不能证明学会语义。

两条真实TRAIN crop（129/97帧）完成12次codec优化，结果见`runs/reaction_program/milestone1_v1/implementation_smoke/RESULTS.json`。总目标2.56680584→3.72581840，两层active code数各2→1；最后AU/VA/expression MSE为0.18521817/0.09465823/0.06190611。**没有收敛或通过codec gate。** 不因脚本成功退出而扩大训练。

Executor仅以明确命名的DIAGNOSTIC_PLACEHOLDER作一次反传检查，CE6.51858759，优化步0；真实pending observation被接口拒绝。该检查不是动作标签训练、不是EXP-C。没有保存新大权重、没有新正式FRC/FRD/diversity成绩。
