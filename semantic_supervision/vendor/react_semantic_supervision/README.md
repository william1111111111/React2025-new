# REACT 语义监督方案包

入口：`CODEX_SEMANTIC_SUPERVISION.md`。
内容：研究/训练方案、标注员说明、四种教师提示词、最小JSON schema、合成示例、元数据与来源校验脚本。

`examples.synthetic.json` 全部是假设示例；没有对用户的任何音频/视频完成标注。
脚本只检查schema、时间范围、声明的角色来源和split；不验证标签语义，也不能替代实际权限隔离。

运行元数据测试：
```bash
python -m unittest -v test_validation.py
python validate_annotations.py examples.synthetic.json --allow-examples
```
环境需要 Python 与 jsonschema。合成示例默认禁止进入训练。

本包没有新模型实现、外部API调用、真实标注或训练成绩。
