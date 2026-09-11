# 单 seed 收尾：实际命令与资源

基线HEAD7e6adf562d628aff78a9e22d74a111def7fa3353，分支agent/hirp-phase25-timescale-alignment。多seed已按用户要求停止；本段没有训练命令。

1. `.venv/bin/python runs/phase25/timescale_v1/final_seed123_v1/numerical_probe.py`（GPU0）：C0/C1/C2×4000/6000×固定case0/20/40/60，FP32/FP64与候选chunk10/3对照。日志numerical_probe.txt，结果numerical_probe.json。
2. `.venv/bin/python -m pytest hirp/tests/test_phase25_final.py -q`：source-serial显存设置与原public sampler显式noise的一致性。tests_sampler.txt。
3. `bash runs/phase25/timescale_v1/final_seed123_v1/run_missing_tasks.sh`，tmux hirp25_finish_tasks（GPU0）。实际子命令：
   - `.venv/bin/python -m hirp.task_phase25_final --checkpoint runs/phase25/timescale_v1/seed_123/C0/attempt_001/checkpoints/step_004000.pt --device cuda:0`
   - `.venv/bin/python -m hirp.task_phase25_final --checkpoint runs/phase25/timescale_v1/seed_123/C0/attempt_001/checkpoints/step_006000.pt --device cuda:0`
   新namespace、不覆盖旧严格检查失败。C0_task4000.txt、C0_task6000.txt和task_multitarget原始结果。
4. `.venv/bin/python -m hirp.evaluate_phase25_final --device cuda:0`，tmux hirp25_finish_es；与第3项并行。T750/T128、三个arm、六checkpoint；bank0学习曲线，2000/6000额外使用既有bank1；共48项。es_stdout.txt、distribution/completed.json及原始逐例结果。没有新noise bank。
5. `.venv/bin/python -m pytest hirp/tests -q`：137passed、3skipped、1xfailed；CPU全仓路径，tests_full.txt。不能称跳过的CUDA/AMP路径已在本段重跑。
6. `.venv/bin/python runs/phase25/timescale_v1/final_seed123_v1/monitor.py`：每30秒只读记录，无异常，完成48/48和2/2后退出；monitor_history.jsonl。
7. `.venv/bin/python runs/phase25/timescale_v1/final_seed123_v1/auxiliary.py`：读取既有最终K10预测与冻结Processor target cache，covariance、lag1/10/30、固定四案例全长轨迹图，无生成。auxiliary_stdout.txt。
8. `.venv/bin/python runs/phase25/timescale_v1/final_seed123_v1/build_report.py`：汇总18任务点、48ES点、两bank端点均值、配对session区间、gradient/saturation轨迹、checkpoint指纹与曲线；analysis_stdout.txt。
9. `.venv/bin/python runs/phase25/timescale_v1/final_seed123_v1/bridge_cost.py`：读取原始数组shape计算旧训练有效帧曝光，核对两协议前2000步source/reference ID调度，生成成本桥接表；bridge_cost_stdout.txt。
10. `git diff --check`：通过。新Python源码此前`py_compile`通过。
11. 本目录verify_delivery.py执行内容hash与历史产物核验，并写入verify_delivery.json及artifact_fingerprints.json。

所有GPU计算均在用户指定GPU0；实际每点评价秒数/显存见analysis/task_resources.csv和ES_all_banks.csv，所有源码、checkpoint与结果内容hash见交付指纹。共享GPU任务的时间不等于独占卡基准性能。

训练/并行调度/CPU exact FRD早先的实际命令见上级EXECUTED_COMMANDS.md、commands/launch_000/*command.json、parallel_budget_v2/*command.json和相关日志。失败、暂停及已完成记录全部保留；不将部分seed42算作完成实验。不push、不提交权重/数组/环境文件。
