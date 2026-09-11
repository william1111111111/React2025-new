# Phase25：时间尺度对齐（本段实际状态）

既有模型的 C2−C1 官方 FRC 六组均为正，但新增目标侧/一对一匹配诊断各只有四组为正；生成侧匹配优势不能自动解释为完整目标池覆盖。下一项决策是安排可用 GPU，执行已经冻结的 T750 seed123 三 arm 首段，而非继续调 lambda。正式长训练尚未执行。

## 改变的训练覆盖条件

保留完整 HiRP22/standard_normal/pre-norm/default projections 和 C0/C1/C2 目标。新协议把 T128 改为 T750，并使用 occurrence-specific source-only 均匀裁剪、独立 reference 裁剪和新的 train-only T750 scaler。因裁剪、长度和总预算均变，未来对比旧结果必须同时报告帧曝光与成本，不能只归因于 dilation。

实际默认宽度 TemporalBlock 与完整 HiRP22 回归确认：T128 时 d128 左右位置数据梯度为零；T750 完整模型对应 L1 范数46.368/41.412，center55.348。测量发生在 optimizer 前，不把 AdamW decay 算成数据梯度。mask前后、K-prefix、public forward/sample/loader 测试通过。这只说明该连接的支持范围。

## 冻结矩阵诊断：C2 minus C1

仍为 Phase24 Development80、K10、原10目标槽位，包括重复目标；未新增生成。官方 FRC 原候选求和，辅助两列为均值，不是成功率或模式数。

| seed | lambda | official FRC差 | target-side mean差 | one-to-one mean差 |
| --- | --- | --- | --- | --- |
| 123 | 0.03 | +0.009730 | +0.003466 | +0.002410 |
| 123 | 0.1 | +0.005525 | -0.000140 | -0.000261 |
| 42 | 0.03 | +0.021546 | +0.000856 | +0.000867 |
| 42 | 0.1 | +0.008354 | +0.001499 | +0.001835 |
| 2026 | 0.03 | +0.019411 | -0.001301 | -0.000786 |
| 2026 | 0.1 | +0.075255 | +0.000602 | +0.001608 |

全部15个HiRP模型与原生归档MAM逐模型/逐输入数据见 `coverage/per_model.csv`、`per_input.csv`；图见 `matching_differences.png`。lambda变体不算额外训练seed。MAM不是等预算参考。

## 已实际运行的训练检验

| arm | CPU smoke实际步数 | paired ES | total | optimizer compute秒 |
| --- | --- | --- | --- | --- |
| C0 | 1 | 0.493688 | 0.493688 | 7.994 |
| C1 | 1 | 0.493688 | 0.753780 | 10.010 |
| C2 | 1 | 0.493688 | 0.724390 | 9.902 |

这三次使用真实训练数据、完整网络、T750/B4/K4/FP32，相同初始化与调度；仅为首步smoke，不能作为学习曲线。真实C2另做连续2步 vs 1+1恢复，模型/优化器最大误差0、训练行完全一致。三arm缩小模型的4步恢复也精确一致。

正式seed123 C0/C1/C2均为0/2000首段步数；预定总预算每arm/seed6000，尚无新checkpoint任务分数、GPU吞吐或显存实测，不以CPU数值填这些列。CPU smoke保存在独立目录，不用于正式热启动。

## 数据与调度契约

三seed完整6000步调度已生成，实际训练source总体1660。各seed24000次occurrence，审计有效source/pair帧分别17135366、17102543、17152632，全无效pair0。source裁剪不读取target长度决定起点；reference独立随机裁剪，C0不读取reference值。只缓存原始数组，避免冻结随机crop。重复读取相同occurrence逐tensor一致。

T750 scaler按p(session)=n_source/N、session内唯一reference均匀、每reference4次固定均匀crop(seed25000)拟合。source-paired有限总体和全reference有限总体仍不同，不虚称完全同一边际。新scaler不可与历史T128 descriptor分数直接算百分比。三seed共享其内容hash，各自初始化/source/reference/noise/crop流独立。每个seed manifest包含完整文件指纹和6000步配置；运行launcher另保存实际源码快照与命令。

## Exact FRD 子集

预固定每session按clip_id排序第一条，共20条；不是全Development80。MAM已完成 2000/2000 candidate-target pairs，完整source 20/20，FRD=172.57642347297096。原版距离：每candidate取10targets最小值后求和；每pair是 AU/15 + VA + expression/8 的三个 unrestricted DTW。不是视频真实性。

首pair1.088秒，后续单线程CPU按pair保存，可复用完整一致记录；小型非等长回归与原版函数误差0。缺损/身份不符缓存报错，不将部分均值当完整成绩。新T750 seed123三模型尚未训练，对应6000pairs未执行。完整80source及其他seed未启动。

## 验证与边界

全仓CPU测试134passed、3skipped、1xfailed（既有CUDA/AMP相关跳过或预期失败）；最后新增“张量T750但有效长度128仍无左右数据梯度”测试；最终10项定向回归通过。2833个Phase24历史产物hash无变化。没有改架构、历史训练函数、checkpoint、旧报告或旧预测；无push。

新 `task_phase25.py` 已接通被冻结的Development80清单、Processor cache、K10/global noise/750 block与原metric实现；因无正式checkpoint，尚未执行新模型任务评价。新增模块做过语法检查，不能称GPU集成已通过。候选分块一致性将在每次真实导出验证。

GPU两次检查均8卡忙；GPU0约18GB/80GB且约59%利用率，共享选择尚待回复。不能把CPU smoke的开销外推为GPU吞吐。没有停止其他用户进程。

人物/interaction/recording权威映射仍未补齐，沿用Phase24人口未决项；未生成confirmation source，反复使用的Development80不叫独立确认。

## 版本与复核

基线/当前HEAD `7e6adf562d628aff78a9e22d74a111def7fa3353`；本地开发分支 `agent/hirp-phase25-timescale-alignment`。本段新增文件尚未提交。
协议 `PROTOCOL.md`，命令及执行状态 `EXECUTED_COMMANDS.md`，数据/梯度/恢复JSON与全部原始日志位于本目录。后续GPU正式首段由 `python -m hirp.run_phase25 --device <approved GPU> --seed 123 --through 2000 --evaluate` 执行，先做显存smoke；6000步总预算不按验证分数改变。
