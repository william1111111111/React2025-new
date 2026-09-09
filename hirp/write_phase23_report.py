"""Render an evidence-linked report from completed Phase23 tables only."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path('runs/phase23/tradeoff_v1')


def main():
    root=ROOT;a=root/'analysis_v1';s=json.loads((a/'summary.json').read_text());means=[r for r in s['mean_std'] if r['K']==32];rows=[r for r in s['final_rows'] if r['K']==32]
    def fmt(r,k):return f"{r[k+'_mean']:.6f} ± {r[k+'_std']:.6f}"
    lines=['# HiRP Phase 2.3 — 条件质量约束下的有限前沿与任务接入','',
    '## 问题与方法','',
    '本轮检验：在受控条件适配质量下，跨输入边际匹配是否比逐输入群体匹配提供更好的群体拟合？模型仍为 Phase 2.2 的 pre-norm/default HiRP22、A0 标准正态先验；未改变网络、先验、mask、条件 Energy Score 或群体估计器。',
    '', '基线：`135220fae7f9018eec8f8119a72b4e2d074f713b`，原分支 `agent/hirp-phase22-replicated-pilot`；开发分支 `agent/hirp-phase23-tradeoff-evaluation`。初始工作树干净。最终提交见 Git HEAD 与交付消息；没有 push。',
    '', '| 方法 | 冻结目标 |', '|---|---|', '| C0 | 随机 K 条轨迹 paired ES |', '| C1 | 同一 paired ES + λ × 逐输入群体 descriptor ES |', '| C2 | 同一 paired ES + λ × session 边际 descriptor ES |',
    '', '条件项使用原25维unit channel scales、pair_lengths和float32距离；群体项使用原75维描述符及train-only frozen scaler，generated descriptor使用source_lengths。零噪声轨迹仅辅助诊断。\n\n每点 T128/K_train4/B4、AdamW lr=1e-4、weight_decay=.01、FP32、2000 个实际优化步。λ∈{.03,.1,.3}；C0=0。复用九组 C0/λ=.1，新增十二组从零训练。每 seed 内初始化、source/reference 调度、噪声与旧实验逐项相同，不同 seed 的哈希不同。每组 requested=actual=JSONL rows=2000；每组 8000 source occurrences，当前 source 总体1660，对应约4.819倍有放回抽样曝光，不是无放回epoch。详见 [预算与算力](analysis_v1/training_budget.csv)、[checkpoint指纹](artifact_fingerprints.json)、[复用清单](reused_checkpoints.json)。',
    '', '## 证据：完整离散前沿','',
    '以下为重复使用的 Development-80、T128、K32、预定两独立 noise bank 的平均，再跨三个训练 seed 计算描述性均值±样本标准差。不是独立确认。工程容差为每 seed 条件 ES≤其 C0×1.01；阈值在看过 Phase 2.2 后提出，本轮新结果前冻结。它不是统计等效标准。',
    '', '| arm | λ | conditional ES ↓ | group descriptor ES ↓ | 满足1%容差的seed |', '|---|---:|---:|---:|---:|']
    for r in means:lines.append(f"| {r['arm']} | {r['lambda_group']:g} | {fmt(r,'conditional_loss')} | {fmt(r,'marginal_loss')} | {r['screen_pass_seeds']}/3 |")
    lines+=['','![各seed前沿](analysis_v1/figures/frontier_per_seed.png)','','连线仅连接实际训练的离散候选，不表示存在插值模型。完整原始值、cross/self、通道指标与方差见 [逐seed CSV](analysis_v1/per_seed_frontier.csv)、[均值/标准差/方差](analysis_v1/frontier_mean_std.csv)、[逐bank CSV](analysis_v1/per_bank_metrics.csv)。','',
    '| seed | arm | λ | conditional ES | 相对C0变化 | group ES |','|---:|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:(x['seed'],x['arm'],x['lambda_group'])):lines.append(f"| {r['seed']} | {r['arm']} | {r['lambda_group']:g} | {r['conditional_loss']:.6f} | {100*r['conditional_relative_delta_C0']:+.3f}% | {r['marginal_loss']:.6f} |")
    lines+=['','### 约束解释与选择','',
    '采用“所有三个 seed 均满足工作容差，再取平均 group ES 最低的离散点”作为保守开发选择规则；该选择不是统计等效检验。选择文件在任何未来确认前封存。']
    for choice in s['selection']:lines.append(f"- {choice['arm']}："+('本轮无所有seed均通过的候选。' if choice['selected_lambda'] is None else f"选择 λ={choice['selected_lambda']:g}。"))
    lines+=['','**关键限制：上述选择仅来自预定主诊断K32，不能称为稳定的条件质量匹配。** K64改变了容差判定：C2 λ=.1由3/3变为2/3，C1 λ=.1由2/3变为3/3。K32下C2 λ=.1各seed的条件容差session-bootstrap区间也都跨过0。严格匹配/统计等效没有建立，开发候选只能暂定，不能据此直接宣称确认优势。', '', '| arm | λ | K32容差通过seed | K64容差通过seed |','|---|---:|---:|---:|']
    for r in means:
        stability=next(x for x in s['mean_std'] if x['arm']==r['arm'] and x['lambda_group']==r['lambda_group'] and x['K']==64)
        lines.append(f"| {r['arm']} | {r['lambda_group']:g} | {r['screen_pass_seeds']}/3 | {stability['screen_pass_seeds']}/3 |")
    lines+=['','新增前沿证据中，C2的group ES在三个λ、三个seed、K32/K64共18个相同配置对比中均低于C1。λ=.03在K32的条件ES也三seed更低，但K64的seed123方向反转；λ=.3对C1的条件/群体改善均保持方向，却并非所有seed均满足相对C0的条件容差。这个结果支持群体拟合优势及更有利的部分离散点，不支持已经完成稳定的“等条件质量”验证。']
    c1=next(r for r in means if r['arm']=='C1' and r['lambda_group']==.1);c2=next(r for r in means if r['arm']=='C2' and r['lambda_group']==.1)
    lines+=['',f"λ=.1 时 C2 相对 C1 的平均 group ES 变化为 {c2['marginal_loss_mean']-c1['marginal_loss_mean']:+.6f}（{100*(c2['marginal_loss_mean']/c1['marginal_loss_mean']-1):+.2f}%），conditional ES 变化为 {c2['conditional_loss_mean']-c1['conditional_loss_mean']:+.6f}。这不能改写为条件质量相同或 C2 全面胜过 C0。不同 λ 的全部结果已保留，未追加搜索或训练长度。",
    '', '相对 C0 容差的逐seed session bootstrap 见 [质量约束不确定性](analysis_v1/quality_screen_uncertainty.csv)。C2−C1、C2−C0 的 conditional/group/shuffle-gap 直接成对检验见 [成对session对照](analysis_v1/paired_session_contrasts.csv)。这些区间条件于既定模型、开发清单、两bank和三置换；20个session标签不保证对应独立参与者，不能替代训练seed重复，更不能把帧、bank或置换当独立训练seed。',
    '', '### 学习曲线与辅助形态','',
    '所有七点×三seed×五检查点均重新计算 bank0/K32 曲线（105个检查点评价）。[学习曲线CSV](analysis_v1/learning_curves_bank0.csv) 和图标题明确为 **bank0 only**；2000步两bank最终均值没有混进曲线。',
    '', '![seed123曲线](analysis_v1/figures/learning_curves_seed123_bank0.png)',
    '', '额外记录 raw-unit covariance 与 lag1/4/16 自相关，不使用开发集拟合尺度，也不加入损失。它们分别反映跨通道协变及时间持续性，而不是已证实的单输入真实条件分布。六个2-of-4 source混合保留原full-session参考池，见 [source敏感性](analysis_v1/source_subset_sensitivity.csv)。这些仅考察固定Development-80内部的有限source选择，不是更大或独立评价人口。',
    '', '| arm | λ | covariance population RMS ↓ | autocorrelation population RMS ↓ | within-input descriptor distance | between-input descriptor distance |','|---|---:|---:|---:|---:|---:|']
    for r in means:lines.append(f"| {r['arm']} | {r['lambda_group']:g} | {r['aux_covariance_population_RMS_mean']:.6f} | {r['aux_autocorrelation_population_RMS_mean']:.6f} | {r['spread_within_input_descriptor_self_mean']:.6f} | {r['spread_between_input_descriptor_distance_mean']:.6f} |")
    lines+=['','self 项是 ES 目标的组成部分，self 增大本身不构成“刷分”证据；group ES 改善也不证明每个输入的条件分布已匹配。辅助统计和配对评价必须一起解释。',
    '', '## 任务接入：完整开发source序列','',
    '现有九个 checkpoint 各导出同一80条source、169,794个有效帧，长度271–6781，K10 bank0。使用现有750-frame padding/chunk规则，同一sample index在每个时间块共用同一全局ε。没有读取target/session进行预测筛选或排序，没有截掉有效帧、降低K或修改指标公式。',
    '', '实际checkpoint在T128/256/750、短有效长度37下 forward/sample/loader/adapter 误差为0；候选分块最大误差2.44e-6（声明容差3e-6）。1001帧时间分块与整段上下文最大差值1.4973，说明时间分块改变条件上下文，绝不把候选分块等价偷换成时间分块等价。T128训练模型能接收长序列不等于长序列质量已充分验证。详见 [接口审计](long_interface_audit.json)。',
    '', '调用仓库既有FRC/FRVar/S-MSE/TLCC及既有temporal S-MSE扩展。仅单个配对listener目标，因此属于 development/full-source 协议，**不是官方十appropriate-target基准或隐藏测试分数**。原target processor权重/配置已锁定并加载；80条配对长度均与source一致，原版直通分支保留目标，长度重采样分支此次未触发。FRC原实现对10个候选求和，TLCC原实现每clip返回首个候选的lag，均未修改。',
    '', '| seed | arm | FRC（原返回值） | FRVar | S-MSE | temporal S-MSE | TLCC |','|---:|---|---:|---:|---:|---:|---:|']
    task=list(csv.DictReader((a/'task_development_full.csv').open()))
    for r in sorted(task,key=lambda r:(int(r['seed']),r['arm'])):lines.append(f"| {r['seed']} | {r['arm']} | {float(r['FRC']):.6f} | {float(r['FRVar']):.6f} | {float(r['smse']):.6f} | {float(r['temporal_smse']):.6f} | {float(r['TLCC']):.3f} |")
    lines+=['','C2相对C1的FRC方向不在三个seed一致；任务接入没有建立全面任务质量优势。所有任务原始结果及导出指纹见 [任务清单](task_development_full/manifest.json)、[逐seed任务CSV](analysis_v1/task_development_full.csv)。四个预固定案例保存完整25通道、所有10条反应及经验样本平均，见 [轨迹清单](fixed_full_trajectories/manifest.json)，不把零噪声轨迹称为均值或中位数。',
    '', '## 机制依据、缓存与人口边界','',
    '同一固定欧氏描述符空间、独立期望配对、同source权重下，E_x S(P_x,Q)=S(Pbar,Q)+¼E_{x,x′}D_E(P_x,P_x′)。精确有限支持核验误差8.33e-17。逐输入目标额外包含条件分布相似性压力，边际目标没有该项；这是目标代数性质，不是新理论首创，也不自动证明观察到的训练变化由此单独造成。任意pair-dependent mask和有限共享噪声估计不必逐batch满足等式。',
    '', '新evaluator使用完整内容身份：checkpoint/实际步数/arm/seed、计划及source/paired/reference指纹、noise/K/置换、scales/scaler、长度/mask/权重及实现版本。身份差异或旧元数据缺失报错；完整重复调用正常复用completed记录。真实内容变更回归及22项缓存/接口测试通过；完整测试120 passed/3 skipped/1 xfailed。跳过为既有CUDA环境检查，预期失败为既有CPU fused Transformer autocast限制；实际GPU长接口单独通过。本轮未重复第一轮初始化/旧head/toy训练。',
    '', '旧九组36个bank/K组合的新本地评价与历史日志的所有聚合数值最大误差0；这不是审查方的独立复现。历史runs逐文件SHA验证见 [保留审计](historical_preservation.json)。代码、日志、命令和checkpoint清单见 [执行记录](EXECUTED_COMMANDS.md)、[运行源码快照](source_snapshots/manifest.json)。',
    '', 'Development-80继续是反复使用的开发集。原录制/人物/角色跨session映射和Camera↔CSV权威crosswalk未得到，字节不同不能排除同源。confirmation候选保持pending、不生成、不运行；其listener标签已在历史full-VAL参考池中，不能称完全未接触数据。详见 [人口契约](evaluation_population_contract.json)。',
    '', '## 下一步与未执行项','',
    '最明确的优势：固定网格全部18个逐seed/λ/K对比中，C2的群体descriptor ES均低于C1。主问题尚未完全回答：条件容差判定对noise Monte Carlo敏感，尚不能声称稳定匹配的条件质量，更没有建立独立任务质量收益。任何确认前先解决身份与特征provenance，冻结开发选择，再一次性评价；本轮不追加λ或步数。',
    '', '- 独立confirmation：未执行，身份/互动映射及独立性尚未获证实。',
    '- 官方十目标或隐藏测试、视频真实感/FRD、外部MAM：未执行；本轮只接通固定单配对开发属性协议，缺少已锁定匹配比较流程/渲染证据。',
    '- 原processor的非等长目标分支：本清单未触发，不能声称已经实数据验证。',
    '- 长序列等预算训练与额外λ：未执行，超出预定十二次新增训练预算。',
    '- Git push：未执行，按用户要求只保留本地提交。','']
    (root/'PHASE23_REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
