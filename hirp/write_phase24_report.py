"""Evidence-first Phase24 report generated after the fixed matrix completes."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path('runs/phase24/evaluation_v1')


def main():
    a=ROOT/'analysis_v1';s=json.loads((a/'summary.json').read_text());means=s['mean_std'];per=s['per_seed'];contrasts=s['contrasts'];task=s['multi_target'];single=s['single_paired']
    group=[r for r in contrasts if r['contrast']=='C2-C1' and r['metric']=='group_delta'];cond=[r for r in contrasts if r['contrast']=='C2-C1' and r['metric']=='conditional_delta']
    negative=sum(r['mean']<0 for r in group);negative_ci=sum(r['MC_CI_high']<0 for r in group)
    def pair_task(l,comparison):
        rows=[]
        for seed in (123,42,2026):
            c2=next(r for r in task if (r['seed'],r['arm'],r['lambda_group'])==(seed,'C2',l));other=next(r for r in task if r['seed']==seed and r['arm']==comparison and (comparison=='C0' or r['lambda_group']==l));rows.append(c2['FRC']-other['FRC'])
        return rows
    task_positive_c1=sum(v>0 for l in (.03,.1) for v in pair_task(l,'C1'))
    task_positive_c0=sum(v>0 for l in (.03,.1) for v in pair_task(l,'C0'))
    lines=['# HiRP Phase 2.4：评价精度与多目标任务对照','',
    f'**任务证据：** 相同多目标开发清单下，C2在两个λ×三个训练seed的{task_positive_c1}/6组对比中FRC高于C1，{task_positive_c0}/6组高于C0。前者支持本轮监督粒度连接的任务价值，后者支持利用群体标签的价值。λ=.1的单配对FRC不具有同样的全seed一致方向。MAM归档模型的同清单FRC为0.810962，仍明显高于本轮HiRP候选；没有声称超越既有方法或得到独立确认。','',
    f'**最有根据的优势：** 八个新独立K64 bank下，C2−C1群体ES的六个逐seed/λ对比中，{negative}/6均值更低，{negative_ci}/6的bank Monte Carlo区间完全低于0。这是监督连接方式的开发分布证据，区间不包含人物抽样或训练seed不确定性。',
    '', '**下一项决策：** 保留固定方法和C2(.03)/C2(.1)两种用途，不再为1%过线增加训练、bank或λ。将当前可支持主张限定为：在这组三seed从零HiRP对照中，C2相对C1的群体ES和多目标FRC方向一致改善；不包装成超越MAM。下一步先确定合法独立确认人口，MAM原生任务差距不能用descriptor ES替代。',
    '', '## 冻结方法与人群','',
    '基准`8d7e788a88b494882bed1f5565807796da5fb071`，本地分支`agent/hirp-phase24-multitarget-evaluation`。开始时工作树干净，无更晚提交或AGENTS指令。所有模型、损失和历史评价函数未修改；本轮训练步数为0。十五个已有checkpoint按metadata核对arm、实际λ、seed、2000实际步、模型/尺度/prior配置。[候选清单](candidate_manifest.json)保留C0、C1/C2的.03/.1×三seed；.3旧前沿完整保存在[历史表](analysis_v1/historical_frontier_all_lambdas.csv)，没有新增该端点评价搜索。',
    '', '仍使用原Development-80、T128 source-centred crop、unit25维paired ES和train-only frozen75维描述符scaler。条件项pair_lengths、描述符source_lengths、float32距离和群体共享noise index排除规则均不变。8个新bank种子24001–24008，每bank K64，公开sampler/eval模式，每bank独立合法U-statistic后再平均；不flatten为伪iid总体。',
    '', '当前方法解释限于同session、同权重、同固定欧氏空间的理想独立期望：E_x S(P_x,Q)=S(Pbar,Q)+¼E D_E(P_x,P_x′)。C1的群体项额外施加条件分布相似性压力；C2通过边际匹配保留条件差异的余地，同时继续接受同一paired ES约束。它不是说输入间越远越好，也不是所有set-set方法共有的惩罚。任务指标用于检验这种连接差异是否具有实际反应价值。','','## 八bank精度与条件—群体对照','',
    '| 方法 | λ | conditional ES：三训练seed均值±SD | group ES：三训练seed均值±SD |','|---|---:|---:|---:|']
    for r in means:lines.append(f"| {r['arm']} | {r['lambda_group']:g} | {r['conditional_ES_mean']:.6f} ± {r['conditional_ES_seed_std']:.6f} | {r['group_ES_mean']:.6f} ± {r['group_ES_seed_std']:.6f} |")
    lines+=['','![八bank前沿](analysis_v1/frontier_MC.png)','','图中区间只表示固定模型/开发集下的bank Monte Carlo误差；三seed SD单独描述训练重复，二者不能相加为人物独立确认。Student-t df7区间基于八个独立bank差值的小样本近似。','',
    '| seed | λ | 对照 | conditional 差值 [95% MC CI] | group 差值 [95% MC CI] |','|---:|---:|---|---:|---:|']
    for seed in (123,42,2026):
        for l in (.03,.1):
            for comp in ('C2-C0','C2-C1'):
                c=next(r for r in contrasts if (r['seed'],r['lambda_group'],r['contrast'],r['metric'])==(seed,l,comp,'conditional_delta'));g=next(r for r in contrasts if (r['seed'],r['lambda_group'],r['contrast'],r['metric'])==(seed,l,comp,'group_delta'))
                def val(r):return f"{r['mean']:+.6f} [{r['MC_CI_low']:+.6f}, {r['MC_CI_high']:+.6f}]"
                lines.append(f'| {seed} | {l:g} | {comp} | {val(c)} | {val(g)} |')
    lines+=['','每bank原始差值与SE见[paired MC表](analysis_v1/paired_MC_intervals.csv)、[逐bank差值](analysis_v1/paired_per_bank.csv)。预定前四/后四bank均完整展示于[分半诊断](analysis_v1/paired_half_diagnostics.csv)，没有择优选半组或追加采样。','',
    '| seed | 方法 | λ | 1%工作容差状态（相对同seed C0） |','|---:|---|---:|---|']
    mapping={'satisfies':'满足（MC区间整体在阈值内）','does_not_satisfy':'不满足（MC区间整体超阈值）','interval_crosses_threshold':'估计区间跨越阈值'}
    for r in per:lines.append(f"| {r['seed']} | {r['arm']} | {r['lambda_group']:g} | {mapping[r['screen_status']]} |")
    lines+=['','容差判定使用每bank的 L_model−1.01 L_C0 差值及其MC区间，仅是工程筛选，不是统计等效。跨阈值不等于方法无效。[逐seed全量表](analysis_v1/per_seed_8bank.csv)同时保留ES cross/self、spread、SE和区间；历史K32/K64点未改写。',
    '', '## 多目标开发任务：与单配对分表','',
    '本地冻结ReactionDataset的test规则为paired置首，再取同session九条alternative；排除paired后池≥9用random.sample，否则random.choices，单条池允许重复paired，不额外去重。实际采用原constructor的视频池及其顺序，仅把该规则用于已用过的VAL Development-80，未使用confirmation source。真实池大小13/14/111/112/113；每输入10个目标，当前清单无不足池情况，短池规则另有临时单元测试及官方AST逐项等价检验。所有目标路径、hash、role/session/length、选择seed、顺序及权重记录在[多目标清单](multitarget_development_manifest.json)。',
    '', 'C0的历史通用checkpoint配置仍存储lambda_group=.1，但C0分支不读取群体项，有效λ=0；候选清单同时记录并校验这两个字段，没有改写checkpoint。','','同一source全长271–6781帧、总169794帧，K10、同一旧bank0前十ε、固定750块；每sample index跨时间块共用ε。全部导出检查长度、有限值、合法域和候选分块一致性；[块边界诊断](analysis_v1/boundary_diagnostics.csv)比较750块边界与普通相邻帧变化，没有新增平滑。MAM保持原生固定集合，没有伪装成随机样本。',
    '', '719个目标非等长。真实短目标958→1926、长目标1926→1167、跨三段2253→1053均通过原Processor；相同RNG下改变预测值而保持shape的最大误差0。原随机重参数化不修改，处理一次后共享同一缓存。目标处理RNG是固定协议条件，未作为另一层随机性重复。[非等长集成记录](unequal_processor_tests.json)。',
    '', '![任务FRC](analysis_v1/task_FRC.png)','','### 多目标表','', '| seed | 方法 | λ | FRC ↑ | FRVar | S-MSE | temporal S-MSE | TLCC（原版） |','|---:|---|---:|---:|---:|---:|---:|---:|']
    def append_task(rows):
        for r in sorted(rows,key=lambda r:(r['arm']=='MAM',r['seed'],r['arm'],r['lambda_group'] or 0)):
            lines.append(f"| {r['seed']} | {r['arm']} | {r['lambda_group'] if r['lambda_group'] is not None else 'native'} | {r['FRC']:.6f} | {r['FRVar']:.6f} | {r['smse']:.6f} | {r['temporal_smse']:.6f} | {r['TLCC']:.3f} |")
    append_task(task)
    lines+=['','### 单配对表（独立诊断口径）','','| seed | 方法 | λ | FRC ↑ | FRVar | S-MSE | temporal S-MSE | TLCC（原版） |','|---:|---|---:|---:|---:|---:|---:|---:|'];append_task(single)
    lines+=['','### 三类证据分别解释','']
    for l in (.03,.1):
        for other in ('C1','C0'):
            vals=pair_task(l,other);lines.append(f"- C2({l:g})−{other}多目标FRC：seed123/42/2026分别为 "+', '.join(f'{v:+.6f}' for v in vals)+('；这是连接粒度对照。' if other=='C1' else '；这是群体标签利用价值对照。'))
    lines+=['','多目标与单配对的对照提示：λ=.1的收益更稳定地体现在群体目标匹配，而不是每seed都提升单个配对轨迹对应关系。这与监督粒度主线一致，但FRC的目标最大匹配不等于完整条件分布恢复；当前人口仍是开发集。','','FRC遵守逐候选在十目标中取最大CCC、再对十候选求和的原实现；没有把候选求和改平均，也没有据GT选择/排序输出。每输入保存10×10候选—目标CCC、候选对距离、通道时间方差和首候选TLCC，足以复核原聚合。TLCC保持原首候选实现（并可在极端lag饱和）；temporal S-MSE明确为仓库既有扩展，不新增指标定义。FRVar/S-MSE不是越大越好的质量结论。',
    '', 'MAM使用归档明确标注的offline evidence代表性checkpoint，经原MANIFEST核验；不是从EQR名字推断。原档说明的/tmp源码目录已不存在，不能恢复精确原git commit；采用完整归档源码/权重hash锁定版本。该模型有预训练anchor50epoch、EQR warmstart、B_fit三epoch，原生AU rounding、residual=.95、style=.95及1.8/.4/1.6通道系数保持不变。本轮公共FP32/TF32关闭（归档CLI默认TF32开启）已记录，不能称原CLI逐bit复现。目标manifest/Processor/长度/K/指标对齐，但训练预算、原生输出后处理和候选语义不同，仅一个MAM seed。因此该行是任务参考，不证明架构或监督粒度的因果优势。[MAM来源](mam_provenance.json)、[实际接入](mam_integration.json)。',
    '', '## 实现与验证附件','',
    '新namespace完整依赖快照覆盖HiRP实际model/data/descriptor/scorer Python和legacy evaluator源文件、Processor权重/config；离线报告/setup/test工具不参与推理依赖。先验签实际mean/std再生成或读取任何导出/结果缓存。旧cache没有补标签或覆盖。临时decoder/session模块及归一化内容变更均拒绝复用；原输入/噪声聚合数值误差0。[缓存回归](cache_dependency_regression.json)、[执行命令](EXECUTED_COMMANDS.md)、[数据与人口未决项](population_pending.json)。',
    '', '源码回归124 passed/3 skipped/1 xfailed，另新增官方AST选择规则的5项focused测试通过。跳过项为既有CUDA环境检查，预期失败为既有CPU autocast限制；真实GPU非等长与模型导出另外执行。没有运行任何新训练、λ=.3扩展或新增noise bank。',
    '', '## 明确未执行与边界','',
    '- 独立confirmation：未执行；缺权威recording/interaction与人物跨session/split映射、Camera↔CSV crosswalk及完整特征provenance。没有用面孔推断身份。历史full-VAL listener pool已参与开发，不能称候选标签未接触。',
    '- exact FRD：已有实现，但本轮未分配16模型×80全长输入×100候选目标对的二次DTW预算；未用descriptor ES或0替代。距离型FRD也不是视频真实感。',
    '- 渲染/视频真实感、隐藏测试或榜单：未运行。本轮只叫“Development-80、官方多目标规则/指标实现对齐的开发评价”。',
    '- 更多训练seed、λ、bank和模型重构：未执行。普通条件随机生成、FiLM、ES、代数分解不作为首次发明；相似性压力解释只适用于当前独立期望/同一固定空间目标，不能泛化到所有set-set方法。',
    '- Git push：未执行；提交只包含代码、协议/指纹、指标和报告，不含数据数组、checkpoint、凭据或运行环境。','']
    (ROOT/'PHASE24_REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
