# BERT 语义受控实验材料

- `CODEX_BERT_SEMANTIC_EXPERIMENTS.md`：完整本地执行任务书。
- `experiment_plan.yaml`：建议配置，不是现有工程可直接执行的训练命令；需由本地 Codex 实现入口。BERT revision 故意留空，必须解析实际不可变版本后填写。
- `reference_ops.py`：上下文token池化、相对时间bias的参考实现。
- `verify_reference_ops.py` / `reference_checks.txt`：本轮实际CPU局部测试和输出。

本轮没有下载或运行BERT权重，没有运行用户模型、标注或训练；参考测试只验证局部计算与方案配置，不证明任何性能改善。

在安装torch和PyYAML的环境中执行：
```sh
python verify_reference_ops.py
```
