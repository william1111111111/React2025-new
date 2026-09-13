from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parent
v=json.loads((R/'verification.json').read_text())
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'
parts=pd.read_csv(R/'decomposition_summary.csv');ccc=pd.read_csv(R/'ccc_summary.csv');scores=pd.read_csv(R/'ccc_frc.csv')
native=[['MAM（原生归档）',.810962318916204,172.57642347297096,.15720370411872864,.058911592],['T0',.8596225763708395,133.87666740682033,.08121564239263535,.05375617742538452],['G0',.977341087330938,139.1408843633149,.0615004226565361,.04645656421780586],['G1',.6780531163513746,131.14359555386798,.09431560337543488,.0398782454431057]]
pd.DataFrame(native,columns=['model','FRC80','exact_FRD20','S_MSE80','FRVar80']).to_csv(R/'native_results.csv',index=False)
s='''# 共享噪声实验审查（CPU，2026-09-13）

基准及实际 HEAD：`979b8a830b1c480d03d868b3883247454dabe077`；开始时工作区干净。执行依据为本目录 `REQUEST.md`。本次只读取已有预测/结果、做 CPU 统计，并修改辅助采样返回字段及测试；没有新生成、训练、rho/seed 搜索、FRD 重算或 push。

**结论：选择“多项同时变化，证据不足以单一归因”。** G1 相对 G0 的 FRC 差异主要体现在时间相关因子；但相对实际起点 T0，Pearson、幅度与均值匹配均有贡献，固定目标后也没有唯一主导项。共享先验确实增加了跨块一致的候选均值差异，但这不能解释为语义模式恢复。保留 T0 为较均衡 Flow 参考、G0 为较高 FRC 参考、G1 为相关先验机制对照。

## 1. 原生成绩完整保留

'''
s+=table(['模型','FRC80 ↑','exact FRD20 ↓','官方 S-MSE80','FRVar80'],[[r[0]]+[f'{x:.9f}' for x in r[1:]] for r in native])
s+='''G0/G1 各完成 2000 新优化步、5716122 有效 endpoint 帧，FRD20 各 2000 配对完成。Development80、K10、seed123；FRD 是固定 20-source 子集，非完整 Development80 FRD。Flow 原生连续 AU，FP64 Euler16 生成后转 FP32 官方指标；MAM 原生 AU 舍入且预算不同。不是隐藏测试、多种子或独立确认结论。

## 2. 精确多样性分解与 block 证据

下表为高精度代数结果，不替换上表 FP32 官方 S-MSE。DC 是每候选的时间均值，slow=P8−DC，fast=Y−P8；尾部 P8 用真实长度。分母均为 K(K−1)TD；每 source 等权，属性组对 all 的贡献权重为 AU 15/25、VA 2/25、expression 8/25。组内值已按组维度归一化，不能直接相加。

'''
s+=table(['模型/组','total','DC','slow','fast'],[[f'{m}/{g}']+[f'{parts[(parts.model==m)&(parts.group==g)].iloc[0][c]:.6f}' for c in ['total','DC','slow','fast']] for m in ['MAM','T0','G0-local','G1-shared'] for g in ['all','AU','VA','expression']])
s+=f'''全部 source/group 的恒等式最大误差 **{v['spread_identity_max_error']:.2e}**。G1 相对 T0：DC +0.02720，slow −0.01328，fast −0.00082，合计 +0.01310。G1 相对 G0：DC +0.02718、slow −0.00705、fast +0.01268；所以 G1/G0 的多样性差异不能全部归于 DC。fast 不直接称为噪声，DC 不直接称为语义模式。

相邻块按相同 source/块编号/真实左右长度配对。共 **68 条有相邻块的 source、185 个块对（每组三份，共555行/模型）**；68 个块对带不足750帧的右尾块，最短1帧。12条短记录没有块对，不计为零。先在 source 内平均块对，再 source 等权。covariance 是跨10候选块均值的逐通道样本协方差（分母9）；r 是逐通道按两块候选方差标准化后的 Pearson，再平均，绝非对平均 covariance 直接命名为相关性。

'''
bs=pd.read_csv(R/'block_summary.csv')
s+=table(['模型/组','covariance','标准化块均值 r'],[[f'{r.model}/{r.group}',f'{r.candidate_covariance:.6f}',f'{r.correlation:.4f}'] for r in bs.itertuples()])
s+=f'''与历史 block CSV 的 covariance 最大重建误差 {v['block_covariance_max_error']:.2e}；每对的长度及差值见 `block_matched.csv`、`block_G1_minus_G0.csv`。以上显示 G1 的候选块均值标准化关联也增加，但不是反应与 GT 的时间相关性。1帧尾块保留、未人为剔除。

四条已有 source（0/20/40/60）探针中，更换 global 的时间均值平均绝对变化为 AU/VA/expression=0.09388/0.09592/0.09367，更换 local 为0.02633/0.01799/0.02024。仅支持这四条的因子敏感性，不外推为全数据语义解耦。

## 3. CCC 精确分解

读取原版 `framework/metrics/FRC.py`：每 candidate 先在原始10×10、25通道均匀平均 CCC 矩阵上选一个 target（并列用首个最大值）；不按通道重选。逐通道使用官方 FP32 mean/var/std、np.cov 的 Pearson、clip[-1,1] 与 eps=1e-8。原25通道分支将常量通道的未定义 Pearson 经 nan_to_num 变为0用于官方 CCC；形态统计中仍标为未定义而非零。预测常量通道比例为0；下表常量比例是目标常量比例，也等于本次未定义 Pearson 比例。

逐通道先验证 CCC=r×2σpσy/(σp²+σy²+(μp−μy)²+eps)，再平均25通道、求和10候选、等权平均80 source。绝不以平均r×平均校准因子替代 CCC。最大逐通道误差 **{v['ccc_identity_max_error']:.2e}**；选定配对相对保存矩阵最大误差 **{v['ccc_matrix_max_error']:.2e}**；A组原生FRC精确还原到显示精度。不是重跑全矩阵或重新选择候选。

下表：均值失配=(μp−μy)²；均值占比=该项/含eps的总分母。σ比为每source内候选/有效通道的中位数，再等权平均source，避免零目标方差导致的无限比值；完整均值及分位数也保留在CSV。r 为每source内有效 candidate-channel 平均，再 source 等权；括号是80个source均值的p10/p50/p90，不把帧或候选视为独立人群。CSV另含每source内r分布分位数。

### A. 原生最佳目标（解释真实官方FRC）

'''
for pairing in ['A_native','B_fixed_G0']:
    if pairing=='B_fixed_G0':s+='### B. 固定 G0 最佳目标（仅诊断，不替换成绩）\n\n'
    q=ccc[(ccc.pairing==pairing)&(ccc.group!='all')]
    s+=table(['模型/组','均值失配','均值占比','σp/σy*','r均值（source p10/p50/p90）','目标常量%'],[[f'{r.model}/{r.group}',f'{r.mean_mismatch:.4f}',f'{r.mean_fraction:.3f}',f'{r.std_ratio_within_q50:.3f}',f'{r.pearson_r:.3f} ({r.pearson_r_source_q10:.3f}/{r.pearson_r_source_q50:.3f}/{r.pearson_r_source_q90:.3f})',f'{100*r.constant_y:.2f}'] for r in q.itertuples()])
s+='''G1 原生目标槽位相对 G0 有62%切换，T0有52.5%切换；这些是槽位切换率，重复GT槽位不等于不同语义模式。B固定的是同一 source、同一 candidate 序号对应的 G0 target；不是新的 GT 噪声选择，也没有修改预测。

'''
s+=table(['配对','模型','重建FRC/诊断分数','GT均值项置零敏感度'],[[r.pairing,r.model,f'{r.reconstructed_FRC:.6f}',f'{r.GT_zero_mean_sensitivity:.6f}'] for r in scores.itertuples()])
s+='''最后一列仅在固定配对的代数表达式中将均值失配项设零，使用GT，未产生新预测，不是可部署成绩、保证上界或真实模型性能。A中G0与G1差距0.29929，均值项同时置零后差距仍为0.32221；B中0.45191变为0.48266。因此均值误差不是充分解释。

进一步令CCC=r×S×M，其中S=2σpσy/(σp²+σy²+eps)，M=(σp²+σy²+eps)/(σp²+σy²+均值项+eps)。对三因子所有6种替换顺序取对称平均，得到精确可加的代数贡献；下表乘10后使用FRC量纲。未定义r仅在官方重建中按0处理，不能把那部分解释为形态r=0；这是描述性分解，不是因果干预证据。

'''
a=pd.read_csv(R/'ccc_attribution_summary.csv');b=pd.read_csv(R/'ccc_G1_minus_T0_summary.csv');attrrows=[]
for pairing in ['A_native','B_fixed_G0']:
    for name,data in [('G1−G0',a[(a.model_minus_G0=='G1-shared')]),('G1−T0',b)]:
        row=data[(data.pairing==pairing)&(data.group=='all')].iloc[0]
        vals=[10*row[c] for c in ['r_contribution','scale_contribution','mean_contribution']]
        attrrows.append([pairing,name]+[f'{x:.6f}' for x in vals]+[f'{sum(vals):.6f}'])
s+=table(['配对','差值','r项','幅度S项','均值M项','合计'],attrrows)
s+='''相对G0，r项占FRC负差约72%（A）/70%（B）；但相对T0，A为约44%/34%/22%，B为约29%/38%/33%。这排除了“只是DC导致均值偏移”的简单解释，也不支持把相对所有基线的损失全部归于时间形态/相位。Pearson本身不能区分相位错位与其他时间形态差异。

## 4. 先验尺度与优化线索

单帧方差保持1、非对角时间协方差为ρ。T帧均值的方差为ρ+(1−ρ)/T，相对独立先验的1/T是1+ρ(T−1)。ρ=.05、T=750时比值**38.45**、标准差比**6.200806**。因此它不是对全部时间模态都很小的“5%扰动”，也不意味着输出均值方差必然变成38.45倍。

21个匹配的已有梯度记录节点（1、100…2000）中，λtask‖g_task‖/‖g_FM‖：G0平均1.55554、中位1.35595、范围[0.75502,2.95271]；G1平均0.44139、中位0.48180、范围[0.13821,0.69488]。FM/task余弦均值分别0.06915/0.02561。固定λtask没有保持实际梯度比例；这只是优化线索，既非因果证明，也未据此重新校准或训练。

## 5. 辅助采样接口及验证

`reaction_flow/shared_model.py`只调整return_aux字段：local_noise返回实际base local；global_noise返回实际base global（ρ0且未提供时为None）；initial_state返回组合一次且mask后的状态；prior_metadata返回ρ及坐标/键语义；predictions、valid_mask、NFE保留。兼容字段noise现在与公开noise参数同为base local。调用者用local_noise/global_noise重放，不能把initial_state传入公开noise参数。跨750帧块重放时显式复用同一global_noise；sample并不暗中持久化记录级global。

没有改模型参数、checkpoint或生产预测路径。CPU测试16项全部通过（`tests.txt`），其中共享接口6项覆盖自动生成基础噪声后的精确重放、ρ0父接口一致、K-prefix、candidate chunk、padding、跨block global一致及初始状态只组合一次。回归用微型CPU模型；历史完整预测和成绩没有因接口修改重算。

## 6. 唯一下一步判断（未实施）

**多因素变化，当前证据不足以归于单一原因。** 不继续调大global noise，也不增加DC奖励。若后续启动一个建模假设，优先验证已有的“配对时间条件路径”想法，以相同预算的T0 continuation作对照，并同时观察时间相关性与幅度/均值校准；这仅是待验证方向，不是声称已经定位到唯一原因。本轮没有实施或启动该方向。

复现：在仓库根目录用CPU运行本目录 `analyze.py` → `compare_t0.py` → `write_report.py`。原文件哈希、原FRC源码哈希、检查误差与范围见 `verification.json`；逐candidate/channel统计见 `ccc_channels.csv`，source级汇总及分布见 `ccc_per_source.csv`、`ccc_summary.csv`；两套配对独立保留。没有改写任何原生结果文件。
'''
(R/'REPORT.md').write_text(s)
