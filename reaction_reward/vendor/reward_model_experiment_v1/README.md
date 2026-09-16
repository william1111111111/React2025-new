# 奖励模型实验设计包

先阅读 CODEX_REACTION_REWARD_MODEL.md；experiment.yaml 是对应的固定初始配置。

reference_contracts.py / test_contracts.py 只用于验证偏好标签、划分和局部数学契约，不包含真实REACT模型训练实现。

运行检查：`python -m unittest -v test_contracts`。需要 numpy、torch、PyYAML。实际输出见 TEST_OUTPUT.txt。

本轮未训练裁判，未执行强化学习，未改动GitHub仓库。阶段B默认为关闭。
